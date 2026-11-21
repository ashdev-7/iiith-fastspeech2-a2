"""Pre-registration 3, part A (analysis/PREREGISTRATION_si_augmentation.md): the confirmatory analysis.

Written and committed before the augmentation results were downloaded.

Inputs: A-B2 runs (default runs_augment/speech/wavlm_spknorm_augspeed), B2 runs (runs_capacity/), and
Results/augmentation_lr_predictions.csv (L1n and A-L1n, written by analysis/augmentation_lr.py).
Output: Results/augmentation_study.md, Results/augmentation_tests.csv.

Primary test: bootstrap over the 200 test words AND the 10 seeds (10 000 resamples, seed 0); the
word-only bootstrap of study 2 is secondary. Holm over HA1-HA3 x 2 directions.

    python analysis/augmentation_study.py [--aug_runs runs_augment] [--b2_runs runs_capacity]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.capacity_study import DIRS, N_BOOT, ROPE, boot, load_neural, verdict  # noqa: E402
from analysis.robustness_checks import holm  # noqa: E402
from common.config import PROJECT_ROOT, RESULTS_DIR  # noqa: E402
from common.data import load_index  # noqa: E402

AUG_VAR, B2_VAR = "wavlm_spknorm_augspeed", "wavlm_spknorm"


def per_word(M: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """clips x seeds correctness -> words x seeds accuracy."""
    M = M.reshape(len(M), -1).astype(float)
    cnt = np.bincount(inv)
    return np.stack([np.bincount(inv, weights=M[:, j]) / cnt for j in range(M.shape[1])], 1)


def boot_ws(terms, inv, rng):
    """Primary bootstrap over words and seeds of a signed sum of model accuracies.
    terms: list of (sign, clips x seeds correctness). Neural terms with the same number of seeds share
    one resampled seed index (seeds paired by id); deterministic terms have one column.
    Returns (estimate, lo, hi, two-sided p) in pp."""
    W = [(s, per_word(M, inv)) for s, M in terms]
    est = sum(s * w.mean() for s, w in W)
    n_w = W[0][1].shape[0]
    out = np.empty(N_BOOT)
    for k in range(N_BOOT):
        wi = rng.integers(0, n_w, n_w)
        seed_idx = {}
        tot = 0.0
        for s, w in W:
            n_s = w.shape[1]
            if n_s not in seed_idx:
                seed_idx[n_s] = rng.integers(0, n_s, n_s)
            tot += s * w[wi][:, seed_idx[n_s]].mean()
        out[k] = tot
    p = min(1.0, 2 * min((out <= 0).mean(), (out >= 0).mean()))
    return 100 * est, 100 * np.percentile(out, 2.5), 100 * np.percentile(out, 97.5), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug_runs", default=str(PROJECT_ROOT / "runs_augment"))
    ap.add_argument("--b2_runs", default=str(PROJECT_ROOT / "runs_capacity"))
    args = ap.parse_args()
    df = load_index()
    word_of = df.set_index("uid").word
    lr = pd.read_csv(RESULTS_DIR / "augmentation_lr_predictions.csv")
    ref = pd.read_csv(RESULTS_DIR / "capacity_lr_predictions.csv").query("model == 'L1n'")

    acc_rows, tests = [], []
    for fold, dname in DIRS.items():
        A, A_ens, uids = load_neural(Path(args.aug_runs), AUG_VAR, fold, df)
        B, B_ens, uids_b = load_neural(Path(args.b2_runs), B2_VAR, fold, df)
        assert A.shape[0] == B.shape[0] == 10 and (uids == uids_b).all(), (A.shape, B.shape)
        lf = lr[lr.fold == fold].set_index("uid")
        L = (lf[lf.model == "L1n"].loc[uids, "pred"] == lf[lf.model == "L1n"].loc[uids, "true"]).values
        AL = (lf[lf.model == "A-L1n"].loc[uids, "pred"] == lf[lf.model == "A-L1n"].loc[uids, "true"]).values
        # consistency: L1n recomputed in the augmentation job == L1n of study 2
        r = ref[ref.fold == fold].set_index("uid").loc[uids]
        same = (lf[lf.model == "L1n"].loc[uids, "pred"].values == r.pred.values).mean()
        words = word_of.loc[uids].values
        _, inv = np.unique(words, return_inverse=True)

        for name, M, ens in (("B2", B, B_ens), ("A-B2", A, A_ens)):
            acc = 100 * M.mean(1)
            acc_rows.append(dict(direction=dname, model=name, seeds=len(M), mean=acc.mean().round(2),
                                 sd=acc.std(ddof=1).round(2), min=acc.min().round(2), max=acc.max().round(2),
                                 ensemble=round(100 * ens.mean(), 2)))
        for name, v in (("L1n", L), ("A-L1n", AL)):
            acc_rows.append(dict(direction=dname, model=name, seeds=1, mean=round(100 * v.mean(), 2)))
        acc_rows[-2]["l1n_identical_to_study2_pct"] = round(100 * same, 2)

        At, Bt = A.T, B.T                                          # clips x seeds
        spec = [
            ("HA1", "A-B2 - B2", "> 0", [(1, At), (-1, Bt)], (At.mean(1) - Bt.mean(1))),
            ("HA2", "A-L1n - L1n", ">= -2 pp", [(1, AL), (-1, L)], (AL.astype(float) - L)),
            ("HA3", "(A-B2 - B2) - (A-L1n - L1n)", "> 0", [(1, At), (-1, Bt), (-1, AL), (1, L)],
             (At.mean(1) - Bt.mean(1)) - (AL.astype(float) - L)),
            ("HA4", "A-B2 - L1n (descriptive)", "none", [(1, At), (-1, L)], (At.mean(1) - L)),
        ]
        for i, (h, comp, pred, terms, delta_clip) in enumerate(spec):
            est, lo, hi, p = boot_ws(terms, inv, np.random.default_rng(0))
            _, wlo, whi, _ = boot(delta_clip, words, np.random.default_rng(0))
            tests.append(dict(hypothesis=h, comparison=comp, direction=dname, delta_pp=round(est, 2),
                              ci_lo=round(lo, 2), ci_hi=round(hi, 2), p_boot=p, prediction=pred,
                              ci_lo_words_only=round(wlo, 2), ci_hi_words_only=round(whi, 2)))
    t = pd.DataFrame(tests)
    conf = t.hypothesis.isin(["HA1", "HA2", "HA3"])
    t.loc[conf, "p_holm"] = holm(t.loc[conf, "p_boot"].values)
    verdicts = {}
    for h, g in t.groupby("hypothesis", sort=False):
        cis = [(r.delta_pp, r.ci_lo, r.ci_hi) for r in g.itertuples()]
        verdicts[h] = verdict(cis)
    t["verdict"] = t.hypothesis.map(verdicts)

    def supported(r):
        if r.prediction == "> 0":
            return "yes" if "replicated (+)" in r.verdict else "no"
        if r.prediction == ">= -2 pp":
            return "yes" if all(g >= -ROPE for g in t[t.hypothesis == r.hypothesis].ci_lo) else "no"
        return "-"
    t["prediction_supported"] = [supported(r) for r in t.itertuples()]
    a = pd.DataFrame(acc_rows)
    t.to_csv(RESULTS_DIR / "augmentation_tests.csv", index=False)
    (RESULTS_DIR / "augmentation_study.md").write_text(
        "# Pre-registered augmentation study (SI)\n\nPlan: `analysis/PREREGISTRATION_si_augmentation.md` "
        "(pushed before the runs). A- = trained on the training speaker's original clips plus 0.9x and 1.1x "
        "speed-perturbed copies (each copy normalised as its own pseudo-speaker). † transductive "
        "normalisation in every row.\n\n## Accuracy per direction\n\n" + a.to_markdown(index=False) +
        "\n\n## Pre-registered tests\n\nPrimary CI = bootstrap over test words and seeds (10 000 resamples); "
        "`*_words_only` = the word-only bootstrap of study 2 (secondary). Holm over HA1-HA3 x 2 directions. "
        f"Verdicts use the primary CIs with a ROPE of +-{ROPE:g} pp.\n\n" + t.to_markdown(index=False) + "\n",
        encoding="utf-8")
    print(a.to_string(index=False))
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
