"""Pre-registration 5 (analysis/PREREGISTRATION_si_final.md): the confirmatory analysis.

Written and committed before the job was run; validated on random (null) and planted-signal features only.

Inputs: CACHE_DIR/pooled_<enc>_<copy>.npy from analysis/final_features.py (enc = wavlm_large, whisper_large).
Outputs (in --results_dir, default Results/): final_study.md, final_scores.csv, final_tests.csv,
final_predictions.csv, final_selftrain_diagnostic.csv.

The labels of the test speaker are used in one place only: `score()`, after every prediction is fixed.
Every model-fitting function receives the training speaker's labels and test-speaker FEATURES only.

    python analysis/final_study.py
    python analysis/final_study.py --validation --results_dir DIR    # null / planted-signal checks only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.backbone_study import audits, zscore  # noqa: E402
from analysis.capacity_inductive import calibration_words  # noqa: E402
from analysis.capacity_study import DIRS, ROPE, boot, verdict  # noqa: E402
from analysis.final_features import pitch_name  # noqa: E402
from analysis.robustness_checks import holm  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from common.audio import PITCH_STEPS, SPEED_FACTORS  # noqa: E402
from common.config import CACHE_DIR, EMOTIONS, RESULTS_DIR  # noqa: E402
from common.data import load_index, make_split  # noqa: E402

C_FIXED = 0.001
FILES = {"WL": "wavlm_large", "W3": "whisper_large"}
SPEED = [f"sp{int(round(100 * f)):03d}" for f in SPEED_FACTORS]
PITCH = [pitch_name(k) for k in PITCH_STEPS]
RECIPES = {"A": SPEED, "AP": SPEED + PITCH}          # training copies added to the original audio
ST_FRACTIONS = (0.25, 0.50, 0.75)                     # self-training rounds: share of the pool pseudo-labelled
C1_MIN = 99.0                                         # % identical predictions to study 4 for A-L1n(WL)


FEAT_DIR = CACHE_DIR                                  # set by --feat_dir


def load_feats(enc: str, copy: str) -> np.ndarray:
    P = np.load(FEAT_DIR / f"pooled_{FILES[enc]}_{copy}.npy")
    return P.reshape(len(P), -1)


def member_data(enc, recipe, tr, te, cal_pos, ytr):
    """Training matrix (original + perturbed copies of the TRAINING rows, each copy z-scored as its own
    pseudo-speaker) and the test speaker's features z-scored two ways: transductive (statistics of all test
    clips, †) and inductive (statistics of the 40 calibration words only). No test label enters here."""
    X = load_feats(enc, "orig")
    parts = [zscore(X, tr, tr)] + [zscore(load_feats(enc, c), tr, tr) for c in RECIPES[recipe]]   # audit D4
    Xaug, yaug = np.concatenate(parts), np.tile(ytr, len(parts))
    return Xaug, yaug, {"T": zscore(X, te, te), "I": zscore(X, te, te[cal_pos])}


def fit_predict(Xa_parts, ya, Xe, C=C_FIXED):
    """The A-L1n classifier (analysis/si_linear.py:clf = StandardScaler + LogisticRegression(C, max_iter=5000)),
    run step by step to save memory (amendment 1): the scaler works in place on a fresh float32 copy, which is then
    converted to float64 exactly as LogisticRegression's lbfgs input check would do. Same arithmetic as clf(C)."""
    X = np.concatenate(Xa_parts)                                # always a fresh copy
    sc = StandardScaler(copy=False).fit(X)
    X = sc.transform(X).astype(np.float64)                      # in-place scaling, then lbfgs's own dtype
    m = LogisticRegression(C=C, max_iter=5000).fit(X, ya)
    del X
    assert list(m.classes_) == list(range(len(EMOTIONS)))
    return m.predict_proba(sc.transform(Xe, copy=True))


def self_train(members, P0, pool):
    """Self-training on unlabelled test-speaker clips. members: [(Xaug, yaug, Xeval)], P0: round-0 mean
    probabilities on Xeval rows, pool: positions (into Xeval) that may be pseudo-labelled. Each round takes the
    pool clips with the highest ensemble confidence (no class quota, no use of any class prior), labels them
    with the ensemble's argmax, and refits every member on its training rows plus those clips (original audio).
    Returns the final probabilities and the per-round (selected positions, pseudo-labels)."""
    P, hist = P0, []
    for f in ST_FRACTIONS:
        k = int(round(f * len(pool)))
        sel = pool[np.argsort(-P[pool].max(1), kind="stable")[:k]]
        yp = P[sel].argmax(1)
        hist.append((sel, yp))
        P = np.mean([fit_predict([Xa, Xe[sel]], np.concatenate([ya, yp]), Xe) for Xa, ya, Xe in members], 0)
    return P, hist


def run_direction(df, fold, calib):
    """All pre-registered models for one direction -> {model: {mode: proba over test rows}}, self-training
    history. mode T = transductive †, I = inductive (40 calibration words; scored on the other 160)."""
    s = make_split(df, "si", 0, fold)
    tr, te = np.concatenate([s["train"], s["val"]]), s["test"]
    ytr = df.label.values[tr]                                   # the only labels this function reads
    cal_pos = np.flatnonzero(np.isin(df.word.values[te], list(calib)))
    pools = {"T": np.arange(len(te)), "I": cal_pos}             # I: the scored 160 words never enter the pool
    probs, hist = {}, {}

    def base(keys):
        data = {k: member_data(*k, tr, te, cal_pos, ytr) for k in keys}
        for (e, r), (Xa, ya, Xe) in data.items():
            probs[f"{r}-L1n({e})"] = {m: fit_predict([Xa], ya, Xe[m]) for m in "TI"}
            print(f"  {DIRS[fold]} {r}-L1n({e}) fitted", flush=True)
        return data

    def st(name, data, start):
        probs[name] = {}
        for m in "TI":
            members = [(Xa, ya, Xe[m]) for Xa, ya, Xe in data.values()]
            probs[name][m], hist[(name, m)] = self_train(members, start[m], pools[m])
        print(f"  {DIRS[fold]} {name} done", flush=True)

    # one encoder/recipe pair in memory at a time, except the two FINAL members (amendment 1: memory)
    data = base([("WL", "A")])
    st("ST(A-L1n(WL))", data, probs["A-L1n(WL)"])
    del data
    base([("W3", "A")])                                       # fitted, then its data is freed
    data = base([("WL", "AP"), ("W3", "AP")])
    probs["ENS2"] = {m: (probs["AP-L1n(WL)"][m] + probs["AP-L1n(W3)"][m]) / 2 for m in "TI"}
    st("FINAL", data, probs["ENS2"])
    del data
    return te, cal_pos, probs, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=str(RESULTS_DIR))
    ap.add_argument("--feat_dir", default=str(CACHE_DIR), help="folder with pooled_<enc>_<copy>.npy")
    ap.add_argument("--validation", action="store_true", help="null-feature validation: C1 is not enforced")
    args = ap.parse_args()
    global FEAT_DIR
    FEAT_DIR = Path(args.feat_dir)
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_index()
    y, words = df.label.values, df.word.values
    calib = calibration_words(df)
    lines = ["# Pre-registered final study (SI)\n",
             "Plan: `analysis/PREREGISTRATION_si_final.md` (pushed before the run). † transductive (test-speaker "
             "statistics and self-training pool from all its unlabelled clips); I = inductive (both from the 40 "
             "calibration words only, scored on the other 160 words).\n", "## Audits\n"] + audits(df)

    res = {fold: run_direction(df, fold, calib) for fold in (0, 1)}
    # D5: in the inductive variant no scored clip was ever pseudo-labelled
    d5 = all(np.isin(np.concatenate([s for s, _ in res[f][3][(n, "I")]]), res[f][1]).all()
             for f in (0, 1) for n in ("ST(A-L1n(WL))", "FINAL"))
    lines.append(f"* D5 (inductive: only calibration-word clips are ever pseudo-labelled): {'PASS' if d5 else 'FAIL'}.")
    lines.append("* D6: PASS by construction: `run_direction()` reads the training speaker's labels only; test "
                 "labels are read in `main()` after all predictions are fixed.")

    # ---- scoring: the first and only place where test-speaker labels are read ----
    rows, preds, correct, diag = [], [], {}, []
    for fold in (0, 1):
        te, cal_pos, probs, hist = res[fold]
        yte = y[te]
        ncal = np.ones(len(te), bool)
        ncal[cal_pos] = False
        for name, P in probs.items():
            pT, pI = P["T"].argmax(1), P["I"].argmax(1)
            correct[(name, fold)] = (pT == yte).astype(float)
            correct[(name + " I", fold)] = (pI == yte).astype(float)
            rows.append(dict(model=name, direction=DIRS[fold], acc=round(100 * (pT == yte).mean(), 2),
                             acc_160=round(100 * (pT == yte)[ncal].mean(), 2),
                             acc_inductive_160=round(100 * (pI == yte)[ncal].mean(), 2)))
            preds += [dict(model=name, fold=fold, uid=u, true=EMOTIONS[t], pred=EMOTIONS[q], pred_inductive=EMOTIONS[r])
                      for u, t, q, r in zip(df.uid.values[te], yte, pT, pI)]
        for (name, m), h in hist.items():                       # POST HOC diagnostic, never used to choose
            for r, (sel, yp) in enumerate(h, 1):
                diag.append(dict(model=name, mode=m, direction=DIRS[fold], round=r, n_pseudo=len(sel),
                                 pseudo_label_acc=round(100 * (yp == yte[sel]).mean(), 2)))
    sc = pd.DataFrame(rows)
    sc = sc.merge(sc.groupby("model", sort=False)[["acc", "acc_inductive_160"]].mean().round(2)
                  .rename(columns={"acc": "mean_of_directions", "acc_inductive_160": "mean_inductive_160"}), on="model")
    pr = pd.DataFrame(preds)

    # C1: A-L1n(WL) reproduces study 4
    ref = pd.read_csv(RESULTS_DIR / "backbone_predictions.csv").query("model == 'A-L1n(WL)'")
    mine = pr[pr.model == "A-L1n(WL)"].merge(ref, on=["fold", "uid"], suffixes=("", "_ref"))
    c1 = 100 * (mine.pred == mine.pred_ref).mean() if len(mine) else float("nan")
    ok = c1 >= C1_MIN
    lines.append(f"* C1 (A-L1n(WL) reproduces study 4, >= {C1_MIN:g} % identical): {'PASS' if ok else 'FAIL'}: "
                 f"{c1:.2f} % identical predictions ({len(mine)} clips).\n")
    if not ok and not args.validation:
        (out / "final_study.md").write_text("\n".join(lines) + "\n\nSTOPPED at C1: no test computed.\n", encoding="utf-8")
        sys.exit("C1 failed: stopping before any test")

    tests = []
    for h, a in (("H1", "FINAL"), ("H2-W", "A-L1n(W3)"), ("H2-P", "AP-L1n(WL)"), ("H2-S", "ST(A-L1n(WL))")):
        for fold in (0, 1):
            te = res[fold][0]
            est, lo, hi, p = boot(correct[(a, fold)] - correct[("A-L1n(WL)", fold)], words[te], np.random.default_rng(0))
            tests.append(dict(hypothesis=h, comparison=f"{a} - A-L1n(WL)", direction=DIRS[fold], delta_pp=round(est, 2),
                              ci_lo=round(lo, 2), ci_hi=round(hi, 2), p_boot=p))
    for fold in (0, 1):
        te, cal_pos = res[fold][0], res[fold][1]
        ncal = np.ones(len(te), bool)
        ncal[cal_pos] = False
        d = (correct[("FINAL I", fold)] - correct[("FINAL", fold)])[ncal]
        est, lo, hi, p = boot(d, words[te][ncal], np.random.default_rng(0))
        tests.append(dict(hypothesis="H3", comparison="FINAL inductive - FINAL transductive (160 words)",
                          direction=DIRS[fold], delta_pp=round(est, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2), p_boot=p))
    t = pd.DataFrame(tests)
    fam = t.hypothesis.str.startswith(("H1", "H2"))
    t.loc[fam, "p_holm"] = holm(t.loc[fam, "p_boot"].values)
    v = {c: verdict([(r.delta_pp, r.ci_lo, r.ci_hi) for r in g.itertuples()]) for c, g in t.groupby("comparison")}
    t["verdict"] = t.comparison.map(v)
    h1 = t[t.hypothesis == "H1"].verdict.iloc[0]
    h3 = bool((t[t.hypothesis == "H3"].ci_lo >= -ROPE).all())
    best = "FINAL" if h1.startswith("replicated (+)") else "A-L1n(WL)"
    dg = pd.DataFrame(diag)
    lines += ["## Accuracy (%)\n", sc.to_markdown(index=False), "",
              "## Pre-registered tests\n",
              "Word-cluster bootstrap (10 000 resamples, seed 0); Holm over H1 + H2 (8 tests); verdicts with a "
              f"ROPE of +-{ROPE:g} pp.\n", t.to_markdown(index=False), "",
              f"* **H1 (primary, FINAL > A-L1n(WL) in both directions): {'SUPPORTED' if best == 'FINAL' else 'NOT SUPPORTED'}** ({h1}).",
              f"* **H3 (inductive FINAL within 2 pp of transductive): {'SUPPORTED' if h3 else 'NOT SUPPORTED'}.**",
              f"* **Best system by the registered reporting rule: {best}.** No other model is promoted, whatever it scores.\n",
              "## POST HOC diagnostic: accuracy of the pseudo-labels per self-training round (reads test labels after "
              "every prediction is fixed; never used to choose)\n", dg.to_markdown(index=False), ""]
    pr.to_csv(out / "final_predictions.csv", index=False)
    sc.to_csv(out / "final_scores.csv", index=False)
    t.to_csv(out / "final_tests.csv", index=False)
    dg.to_csv(out / "final_selftrain_diagnostic.csv", index=False)
    (out / "final_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
