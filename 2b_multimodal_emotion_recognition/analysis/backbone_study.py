"""Pre-registration 4 (analysis/PREREGISTRATION_si_backbones.md): the confirmatory analysis.

Written and committed before the job was run; validated on random (null) features only.

Inputs: WavLM-Base+ stores wavlm, wavlm_sp090, wavlm_sp110 (encoder B) and CACHE_DIR/pooled_<enc>_<copy>.npy
from analysis/backbone_features.py (WL, HL, E and the leakage control LC).
Outputs: Results/backbone_study.md, backbone_scores.csv, backbone_tests.csv, backbone_predictions.csv
         (+ backbone_c_sensitivity.csv with --c_sensitivity: POST HOC, diagnostic only).

    python analysis/backbone_study.py [--c_sensitivity]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.capacity_inductive import calibration_words  # noqa: E402
from analysis.capacity_study import DIRS, ROPE, boot, verdict  # noqa: E402
from analysis.robustness_checks import holm  # noqa: E402
from analysis.si_linear import clf  # noqa: E402
from common.audio import SPEED_FACTORS, speed_store_name  # noqa: E402
from common.config import CACHE_DIR, EMOTIONS, RESULTS_DIR, SPEAKERS  # noqa: E402
from common.data import load_index, make_split  # noqa: E402

C_FIXED = 0.001
MEMBERS = ["B", "WL", "HL", "E"]                    # ENS members (fixed in the pre-registration)
CONTROL = "LC"                                      # emotion2vec+ seed: leakage control, never a candidate
FILES = {"WL": "wavlm_large", "HL": "hubert_large", "E": "e2v_base", "LC": "e2v_plus_seed"}
COPIES = ["orig"] + [f"sp{int(round(100 * f)):03d}" for f in SPEED_FACTORS]
LC_MAP = {"angry": "angry", "disgusted": "disgust", "fearful": "fear", "happy": "happy", "neutral": "neutral",
          "surprised": "ps", "sad": "sad"}
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def load_feats(enc: str, copy: str) -> np.ndarray:
    """(clips, features) pooled mean+std of every kept layer."""
    if enc == "B":
        from analysis.augmentation_lr import flat
        return flat("wavlm" if copy == "orig" else speed_store_name(int(copy[2:]) / 100))
    P = np.load(CACHE_DIR / f"pooled_{FILES[enc]}_{copy}.npy")
    return P.reshape(len(P), -1)


def zscore(X: np.ndarray, rows: np.ndarray, stats_rows: np.ndarray) -> np.ndarray:
    """Normalise X[rows] with mean/std of X[stats_rows]. Features only: no label is passed in (audit D3)."""
    m, s = X[stats_rows].mean(0), X[stats_rows].std(0)
    return (X[rows] - m) / (s + 1e-6)


def audits(df) -> list[str]:
    out = []
    g = df.groupby(["speaker", "word"]).emotion
    ok = (g.nunique() == len(EMOTIONS)).all() and (g.size() == len(EMOTIONS)).all()
    out.append(f"* D1 (word identity carries no label information): {'PASS' if ok else 'FAIL'}: every "
               f"(speaker, word) pair has each of the {len(EMOTIONS)} emotions exactly once "
               f"({df.groupby(['speaker', 'word']).ngroups} pairs).")
    for fold in (0, 1):
        s = make_split(df, "si", 0, fold)
        tr, te = np.concatenate([s["train"], s["val"]]), s["test"]
        ok = (set(df.speaker.values[tr]) == {SPEAKERS[fold]} and set(df.speaker.values[te]) == {SPEAKERS[1 - fold]}
              and not set(df.uid.values[tr]) & set(df.uid.values[te]))
        out.append(f"* D2 ({DIRS[fold]}): {'PASS' if ok else 'FAIL'}: training rows only {SPEAKERS[fold]}, "
                   f"test rows only {SPEAKERS[1 - fold]}, no shared clip.")
    out.append("* D3: PASS by construction: `zscore()` receives features and row indices only.")
    out.append("* D4: PASS by construction: perturbed copies are indexed with training rows only.")
    return out


def fit_all(df, enc, calib, C=C_FIXED, plain=True):
    """A-L1n (and L1n) for one encoder, both directions -> {model: {fold: (proba_test, proba_test_inductive)}}."""
    y = df.label.values
    X = load_feats(enc, "orig")
    Xc = {c: load_feats(enc, c) for c in COPIES[1:]}
    out = {f"A-L1n({enc})": {}, f"L1n({enc})": {}} if plain else {f"A-L1n({enc})": {}}
    for fold in (0, 1):
        s = make_split(df, "si", 0, fold)
        tr, te = np.concatenate([s["train"], s["val"]]), s["test"]
        cal = te[np.isin(df.word.values[te], list(calib))]
        Xtr = zscore(X, tr, tr)
        Xte = zscore(X, te, te)                                   # transductive †
        Xte_ind = zscore(X, te, cal)                              # inductive: 40 calibration words only
        Xaug = np.concatenate([Xtr] + [zscore(Xc[c], tr, tr) for c in Xc])
        fits = {f"A-L1n({enc})": (Xaug, np.tile(y[tr], 1 + len(Xc)))}
        if plain:
            fits[f"L1n({enc})"] = (Xtr, y[tr])
        for name, (A, b) in fits.items():
            m = clf(C).fit(A, b)
            assert list(m.classes_) == list(range(len(EMOTIONS)))
            out[name][fold] = (m.predict_proba(Xte), m.predict_proba(Xte_ind))
        print(f"{enc} {DIRS[fold]}: " + ", ".join(
            f"{n} {100 * (out[n][fold][0].argmax(1) == y[te]).mean():.2f}" for n in fits), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_sensitivity", action="store_true")
    args = ap.parse_args()
    df = load_index()
    y, words = df.label.values, df.word.values
    calib = calibration_words(df)
    lines = ["# Pre-registered encoder study (SI)\n",
             "Plan: `analysis/PREREGISTRATION_si_backbones.md` (pushed before the run). † transductive "
             "per-speaker normalisation; I = inductive (test-speaker statistics from 40 calibration words "
             "only, scored on the other 160 words).\n", "## Audits\n"] + audits(df)

    probs = {}
    for enc in MEMBERS + [CONTROL]:
        probs.update(fit_all(df, enc, calib))
    probs["ENS"] = {fold: tuple(np.mean([probs[f"A-L1n({e})"][fold][k] for e in MEMBERS], 0) for k in (0, 1))
                    for fold in (0, 1)}

    # leakage control LC-a: emotion2vec+ seed's own head, no training on our side
    zs = np.load(CACHE_DIR / "e2v_plus_seed_zeroshot.npy")
    labs = (CACHE_DIR / "e2v_plus_seed_zeroshot_labels.txt").read_text(encoding="utf-8").splitlines()
    cols = [next(i for i, l in enumerate(labs) if l.split("/")[-1] == en) for en in LC_MAP]
    tess = [EMOTIONS.index(LC_MAP[en]) for en in LC_MAP]
    lc_pred = np.array(tess)[zs[:, cols].argmax(1)]

    rows, preds, correct = [], [], {}
    for fold in (0, 1):
        s = make_split(df, "si", 0, fold)
        te = s["test"]
        ncal = ~np.isin(words[te], list(calib))
        for name, P in probs.items():
            p, p_ind = P[fold][0].argmax(1), P[fold][1].argmax(1)
            c = (p == y[te]).astype(float)
            correct[(name, fold)] = c
            correct[(name + " I", fold)] = (p_ind == y[te]).astype(float)
            rows.append(dict(model=name, direction=DIRS[fold], acc=round(100 * c.mean(), 2),
                             acc_160=round(100 * c[ncal].mean(), 2),
                             acc_inductive_160=round(100 * (p_ind == y[te])[ncal].mean(), 2)))
            preds += [dict(model=name, fold=fold, uid=u, true=EMOTIONS[t], pred=EMOTIONS[q])
                      for u, t, q in zip(df.uid.values[te], y[te], p)]
        c = (lc_pred[te] == y[te]).astype(float)
        correct[("LC-a zero-shot", fold)] = c
        rows.append(dict(model="LC-a zero-shot (emotion2vec+ seed head)", direction=DIRS[fold],
                         acc=round(100 * c.mean(), 2)))
    sc = pd.DataFrame(rows)
    mean = sc.groupby("model", sort=False)[["acc"]].mean().round(2).rename(columns={"acc": "mean_of_directions"})
    sc = sc.merge(mean, on="model")

    # C1: B's A-L1n reproduces study 3
    ref = pd.read_csv(RESULTS_DIR / "augmentation_lr_predictions.csv").query("model == 'A-L1n'")
    pr = pd.DataFrame(preds)
    mine = pr[pr.model == "A-L1n(B)"].merge(ref, on=["fold", "uid"], suffixes=("", "_ref"))
    c1 = 100 * (mine.pred == mine.pred_ref).mean() if len(mine) else float("nan")
    lines.append(f"* C1 (A-L1n(B) reproduces study 3): {'PASS' if c1 == 100 else 'FAIL'}: "
                 f"{c1:.2f} % identical predictions ({len(mine)} clips).\n")

    # hypotheses
    tests = []
    for h, a, b in [("HB1", "ENS", "A-L1n(B)")] + [("HB2", f"A-L1n({e})", "A-L1n(B)") for e in MEMBERS[1:]]:
        for fold in (0, 1):
            te = make_split(df, "si", 0, fold)["test"]
            est, lo, hi, p = boot(correct[(a, fold)] - correct[(b, fold)], words[te], np.random.default_rng(0))
            tests.append(dict(hypothesis=h, comparison=f"{a} - {b}", direction=DIRS[fold], delta_pp=round(est, 2),
                              ci_lo=round(lo, 2), ci_hi=round(hi, 2), p_boot=p))
    for fold in (0, 1):
        te = make_split(df, "si", 0, fold)["test"]
        ncal = ~np.isin(words[te], list(calib))
        d = (correct[("ENS I", fold)] - correct[("ENS", fold)])[ncal]
        est, lo, hi, p = boot(d, words[te][ncal], np.random.default_rng(0))
        tests.append(dict(hypothesis="HB3", comparison="ENS inductive - ENS transductive (160 words)",
                          direction=DIRS[fold], delta_pp=round(est, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                          p_boot=p))
    t = pd.DataFrame(tests)
    fam = t.hypothesis.isin(["HB1", "HB2"])
    t.loc[fam, "p_holm"] = holm(t.loc[fam, "p_boot"].values)
    v = {c: verdict([(r.delta_pp, r.ci_lo, r.ci_hi) for r in g.itertuples()]) for c, g in t.groupby("comparison")}
    t["verdict"] = t.comparison.map(v)
    hb1 = t[t.hypothesis == "HB1"].verdict.iloc[0]
    hb3 = all(t[t.hypothesis == "HB3"].ci_lo >= -ROPE)
    lines += ["## Accuracy (%)\n", sc.to_markdown(index=False), "",
              "## Pre-registered tests\n",
              "Word-cluster bootstrap (10 000 resamples, seed 0); Holm over HB1 + HB2 (8 tests); verdicts with a "
              f"ROPE of +-{ROPE:g} pp.\n", t.to_markdown(index=False), "",
              f"* **HB1 (primary, ENS > A-L1n(B) in both directions): {'SUPPORTED' if hb1.startswith('replicated (+)') else 'NOT SUPPORTED'}** ({hb1}).",
              f"* **HB3 (inductive ENS within 2 pp of transductive): {'SUPPORTED' if hb3 else 'NOT SUPPORTED'}.**",
              "* LC (emotion2vec+ seed) is a leakage control, never a candidate; its rows only show what label "
              "exposure that may include TESS produces.\n"]
    pr.to_csv(RESULTS_DIR / "backbone_predictions.csv", index=False)
    sc.to_csv(RESULTS_DIR / "backbone_scores.csv", index=False)
    t.to_csv(RESULTS_DIR / "backbone_tests.csv", index=False)

    if args.c_sensitivity:
        cs = []
        for enc in MEMBERS + [CONTROL]:
            for C in C_GRID:
                P = fit_all(df, enc, calib, C=C, plain=False)[f"A-L1n({enc})"]
                acc = [100 * (P[f][0].argmax(1) == y[make_split(df, "si", 0, f)["test"]]).mean() for f in (0, 1)]
                cs.append(dict(model=f"A-L1n({enc})", C=C, **{DIRS[f]: round(acc[f], 2) for f in (0, 1)},
                               mean=round(float(np.mean(acc)), 2)))
        cs = pd.DataFrame(cs)
        cs.to_csv(RESULTS_DIR / "backbone_c_sensitivity.csv", index=False)
        lines += ["## POST HOC: accuracy against C (diagnostic only; reads the test speaker; never used to choose)\n",
                  cs.to_markdown(index=False), ""]
    (RESULTS_DIR / "backbone_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
