"""Pre-registration 2 (analysis/PREREGISTRATION_si_capacity.md): the confirmatory analysis.

Reads the capacity-study runs (default runs_capacity/, 10 SI seeds per direction for B1-B4), the
logistic-regression predictions and the inductive-normalisation results written by
analysis/capacity_inductive.py, and produces Results/capacity_study.md with:
  * per-direction accuracy, seed SD/range and seed-ensemble accuracy for every variant;
  * a word-cluster bootstrap (10 000 resamples of the 200 test words, seed 0) 95 % CI for every
    pre-registered paired difference, a verdict (replicated / direction-specific / equivalent /
    inconclusive, ROPE +-2 pp) and Holm-corrected bootstrap p-values;
  * a reproducibility check: retrained seeds 0-2 of B1/B2 vs. the original runs in runs/.

    python analysis/capacity_study.py [--runs runs_capacity]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.robustness_checks import holm  # noqa: E402
from common.config import EMOTIONS, PROJECT_ROOT, RESULTS_DIR, SPEAKERS  # noqa: E402
from common.data import load_index  # noqa: E402

ROPE = 2.0
N_BOOT = 10_000
VARIANTS = {"B1": "wavlm", "B2": "wavlm_spknorm", "B3": "wavlm_spknorm_d64_do0.5_wd0.05",
            "B4": "wavlm_spknorm_attnpool"}
DIRS = {0: "OAF->YAF", 1: "YAF->OAF"}


def load_neural(runs: Path, var: str, fold: int, df):
    """-> (seeds x 1400 correctness matrix, ensemble correctness, uid order)"""
    files = sorted((runs / "speech" / var).glob(f"si_fold{fold}_seed*/test_predictions.csv"),
                   key=lambda f: int(f.parent.name.split("seed")[1]))
    ts = [pd.read_csv(f).set_index("uid") for f in files]
    uids = ts[0].index
    probs = np.stack([t.loc[uids, [f"p_{e}" for e in EMOTIONS]].values for t in ts])
    y = np.array([EMOTIONS.index(e) for e in ts[0].loc[uids, "true"]])
    correct = probs.argmax(2) == y[None]
    ens = probs.mean(0).argmax(1) == y
    return correct, ens, uids


def boot(delta_clip: np.ndarray, words: np.ndarray, rng) -> tuple[float, float, float, float]:
    """Word-cluster bootstrap of a per-clip difference vector -> (mean, lo, hi, two-sided p)."""
    uw, inv = np.unique(words, return_inverse=True)
    per_word = np.bincount(inv, weights=delta_clip) / np.bincount(inv)
    idx = rng.integers(0, len(uw), size=(N_BOOT, len(uw)))
    bs = per_word[idx].mean(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return 100 * per_word.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), min(1.0, p)


def boot2(A: pd.DataFrame | pd.Series, B: pd.DataFrame | pd.Series, words: np.ndarray, rng):
    """POST-HOC (not in the pre-registration): resample words AND seeds. A/B: clips x seeds correctness
    (a Series = deterministic model). Seeds are resampled jointly by seed id, so seed-to-seed training
    noise enters the interval. Returns (lo, hi) of B - A in pp."""
    uw, inv = np.unique(words, return_inverse=True)
    def per_word(M):
        M = M.to_frame() if isinstance(M, pd.Series) else M
        return np.stack([np.bincount(inv, weights=M.values[:, j]) / np.bincount(inv) for j in range(M.shape[1])], 1)
    wa, wb = per_word(A), per_word(B)                     # words x seeds
    out = np.empty(N_BOOT)
    for k in range(N_BOOT):
        wi = rng.integers(0, len(uw), len(uw))
        sa = rng.integers(0, wa.shape[1], wa.shape[1])
        sb = sa if wb.shape[1] == wa.shape[1] else rng.integers(0, wb.shape[1], wb.shape[1])
        out[k] = wb[wi][:, sb].mean() - wa[wi][:, sa].mean()
    return 100 * np.percentile(out, 2.5), 100 * np.percentile(out, 97.5)


def verdict(cis):
    (m0, l0, h0), (m1, l1, h1) = cis
    sig = [l > 0 or h < 0 for l, h in ((l0, h0), (l1, h1))]
    pos = [l > 0 for l in (l0, l1)]
    if all(sig) and pos[0] == pos[1]:
        v = "replicated (" + ("+" if pos[0] else "-") + ")"
        if max(abs(l0), abs(h0), abs(l1), abs(h1)) <= ROPE:
            v += ", but inside the ROPE"
        return v
    if all(-ROPE <= x <= ROPE for x in (l0, h0, l1, h1)):
        return "equivalent (+-2 pp)"
    if sum(sig) == 1:
        return "direction-specific"
    if all(sig):
        return "inconclusive (opposite signs)"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(PROJECT_ROOT / "runs_capacity"))
    runs = Path(ap.parse_args().runs)
    df = load_index().set_index("uid")
    rng = np.random.default_rng(0)
    rng2 = np.random.default_rng(1)
    lr = pd.read_csv(RESULTS_DIR / "capacity_lr_predictions.csv")

    # ------------------------------------------------ per-variant scores (per direction)
    corr, ens, rows, seedmat = {}, {}, [], {}
    for fold in (0, 1):
        for vid, var in VARIANTS.items():
            c, e, uids = load_neural(runs, var, fold, df)
            corr[(vid, fold)], ens[(vid, fold)] = pd.Series(c.mean(0), uids), pd.Series(e.astype(float), uids)
            seedmat[(vid, fold)] = pd.DataFrame(c.T.astype(float), index=uids)   # clips x seeds
            accs = 100 * c.mean(1)
            rows.append(dict(direction=DIRS[fold], model=vid, run_dir=var, seeds=len(accs), mean=accs.mean(),
                             sd=accs.std(ddof=1), min=accs.min(), max=accs.max(), ensemble=100 * e.mean()))
        for vid in ("L1", "L1n", "L2n"):
            t = lr[(lr.model == vid) & (lr.fold == fold)].set_index("uid")
            corr[(vid, fold)] = (t.pred == t.true).astype(float)
            rows.append(dict(direction=DIRS[fold], model=vid, run_dir="logistic regression", seeds=1,
                             mean=100 * corr[(vid, fold)].mean(), sd=np.nan, min=np.nan, max=np.nan, ensemble=np.nan))
    scores = pd.DataFrame(rows)

    # ------------------------------------------------ pre-registered comparisons
    def cmp(a, b, fold, sub=None):
        x, y = corr[b] if isinstance(b, tuple) else corr[(b, fold)], corr[a] if isinstance(a, tuple) else corr[(a, fold)]
        u = y.index if sub is None else y.index[sub(y.index)]
        return (x.loc[u] - y.loc[u]).values, df.loc[u, "word"].values

    tests = []
    calib = set(pd.read_csv(RESULTS_DIR / "capacity_calibration_words.csv").calibration_word)
    for h, a, b, pred, rule in (
            ("H1 (partly blind)", "B1", "B2", "> 0", "pos"),
            ("H2", "B2", "L1n", ">= -2 pp", "ge-2"),
            ("H3", "B2", "B3", "> 0", "pos"),
            ("H4", "B3", "B4", ">= -2 pp", "ge-2"),
            ("H5", "B2", "E(B2)", "> 0", "pos"),
            ("H6", "L1n", "I(L1n)", "within +-2 pp", "rope"),
            ("H7", "L1n", "L2n", "<= 0", "le0")):
        cis = []
        for fold in (0, 1):
            if b == "E(B2)":
                d = (ens[("B2", fold)] - corr[("B2", fold)]).values
                w = df.loc[corr[("B2", fold)].index, "word"].values
            elif b == "I(L1n)":
                t = lr[(lr.model == "I(L1n)") & (lr.fold == fold)].set_index("uid")
                ok_i = (t.pred == t.true).astype(float)
                keep = ~t.calibration_word.values.astype(bool)
                u = t.index[keep]
                d = (ok_i.loc[u] - corr[("L1n", fold)].loc[u]).values
                w = df.loc[u, "word"].values
            else:
                d, w = cmp(a, b, fold)
            m, lo, hi, p = boot(d, w, rng)
            cis.append((m, lo, hi))
            lo2 = hi2 = np.nan
            if b not in ("E(B2)", "I(L1n)"):
                A = seedmat.get((a, fold), corr[(a, fold)])
                Bm = seedmat.get((b, fold), corr[(b, fold)])
                Bm = Bm.loc[A.index]
                lo2, hi2 = boot2(A, Bm, df.loc[A.index, "word"].values, rng2)
            tests.append(dict(hypothesis=h, comparison=f"{b} - {a}", direction=DIRS[fold], delta_pp=m,
                              ci_lo=lo, ci_hi=hi, p_boot=p, prediction=pred,
                              ci2_lo_posthoc=lo2, ci2_hi_posthoc=hi2))
        v = verdict(cis)
        ok = {"pos": all(lo > 0 for _, lo, _ in cis), "ge-2": all(lo >= -ROPE for _, lo, _ in cis),
              "rope": all(-ROPE <= lo and hi <= ROPE for _, lo, hi in cis),
              "le0": all(lo <= 0 for _, lo, _ in cis)}[rule]
        for t_ in tests[-2:]:
            t_.update(verdict=v, prediction_supported="yes" if ok else "no")
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests.p_boot.values)

    # ------------------------------------------------ neural inductive normalisation (B2)
    neur = RESULTS_DIR / "capacity_inductive_neural.csv"
    neur_txt = []
    if neur.exists():
        n = pd.read_csv(neur).pivot_table(index=["fold", "seed"], columns="stats", values="accuracy")
        for fold in (0, 1):
            g = 100 * n.loc[fold]
            neur_txt.append(f"| {DIRS[fold]} | {g.transductive.mean():.2f} | {g.inductive.mean():.2f} | "
                            f"{(g.inductive - g.transductive).mean():+.2f} | {len(g)} |")

    # ------------------------------------------------ reproducibility of retrained seeds 0-2
    rep = []
    for vid in ("B1", "B2"):
        for f in sorted((PROJECT_ROOT / "runs" / "speech" / VARIANTS[vid]).glob("si_fold*_seed*/test_predictions.csv")):
            g = runs / "speech" / VARIANTS[vid] / f.parent.name / "test_predictions.csv"
            if g.exists():
                a, b = pd.read_csv(f), pd.read_csv(g)
                rep.append(dict(model=vid, run=f.parent.name, original=100 * (a.pred == a.true).mean(),
                                retrained=100 * (b.pred == b.true).mean(),
                                same_prediction_pct=100 * (a.pred.values == b.pred.values).mean()))
    rep = pd.DataFrame(rep)

    scores.round(2).to_csv(RESULTS_DIR / "capacity_scores.csv", index=False)
    tests.to_csv(RESULTS_DIR / "capacity_tests.csv", index=False)
    md = ["# Pre-registered head-capacity study (SI)", "",
          "Plan: `analysis/PREREGISTRATION_si_capacity.md` (committed before the runs). "
          "B1 = BiGRU (main), B2 = BiGRU + speaker norm †, B3 = small BiGRU + norm †, "
          "B4 = attentive statistics pooling + norm †, L1/L1n = logistic regression (mean+std) without/with "
          "norm †, L2n = logistic regression (mean only) + norm †. † transductive.", "",
          "## Accuracy per direction (10 seeds for neural heads)", "",
          scores.round(2).to_markdown(index=False), "",
          "## Pre-registered comparisons (word-cluster bootstrap, 10 000 resamples; ROPE +-2 pp; "
          "Holm over all 14 tests)", "",
          tests.round({"delta_pp": 2, "ci_lo": 2, "ci_hi": 2, "p_boot": 4, "p_holm": 4, "ci2_lo_posthoc": 2,
                       "ci2_hi_posthoc": 2}).to_markdown(index=False), "",
          "`ci_lo/ci_hi` = the pre-registered word bootstrap. It treats the 10 seeds as fixed, so it measures "
          "uncertainty over test words only. `ci2_*_posthoc` = **post-hoc** (not pre-registered) bootstrap "
          "over words AND seeds, which also includes training noise; it is the more honest interval for a "
          "claim about the model rather than these 10 trained instances.", ""]
    if neur_txt:
        md += ["## Inductive normalisation of the BiGRU (B2): 40 calibration words, scored on the other 160", "",
               "| direction | transductive | inductive | delta | seeds |", "|---|---|---|---|---|", *neur_txt, ""]
    if len(rep):
        md += ["## Reproducibility: retrained seeds 0-2 vs. the original runs", "",
               rep.round(2).to_markdown(index=False), ""]
    (RESULTS_DIR / "capacity_study.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
