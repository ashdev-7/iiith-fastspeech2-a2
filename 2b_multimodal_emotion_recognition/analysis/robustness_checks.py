"""Adversarial checks of the statistical claims in the report (what a sceptical reviewer would run).

1. Multiple comparisons: Holm-Bonferroni over the paired run-level tests.
2. Effective sample size under SI: the 6 runs are 2 transfer directions x 3 seeds, and the direction
   dominates the variance. Each comparison is repeated per direction (does the sign agree in both?).
3. Regression to the mean in "fusion helps when speech is uncertain": binning clips by model A's
   confidence makes ANY second model look better in A's low-confidence bins. Control: bin by the
   speech model of seed s and compare it with the speech model of another seed s' (same direction).
   The fusion effect is only real to the extent that it exceeds this speech-vs-speech control.

    python analysis/robustness_checks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.config import EMOTIONS, PROJECT_ROOT, RESULTS_DIR  # noqa: E402

RUNS = PROJECT_ROOT / "runs"
BINS = [0, 0.5, 0.7, 0.9, 1.0001]
BIN_LABELS = ["<0.5", "0.5-0.7", "0.7-0.9", ">=0.9"]


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def load(pipeline: str, variant: str, run: str) -> pd.DataFrame:
    t = pd.read_csv(RUNS / pipeline / variant / run / "test_predictions.csv")
    t["ok"] = t.pred == t.true
    t["conf"] = t[[f"p_{e}" for e in EMOTIONS]].max(1)
    return t.set_index("uid")


def run_acc(pipeline: str, variant: str, fold: int) -> list[float]:
    return [load(pipeline, variant, f"si_fold{fold}_seed{s}").ok.mean() for s in range(3)]


def binned_gain(ref: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    m = ref[["ok", "conf"]].join(other[["ok"]], rsuffix="_other")
    m["bin"] = pd.cut(m.conf, BINS, right=False, labels=BIN_LABELS)
    return m.groupby("bin", observed=False).agg(n=("ok", "size"), ref=("ok", "mean"), other=("ok_other", "mean"))


def main():
    out = ["# Robustness checks of the statistical claims", ""]

    # 1) Holm-Bonferroni
    sig = pd.read_csv(RESULTS_DIR / "significance.csv")
    for proto, g in sig.groupby("protocol", sort=False):
        sig.loc[g.index, "p_t_holm_within_protocol"] = holm(g.p_paired_t.values)
    sig["p_t_holm_all"] = holm(sig.p_paired_t.values)
    sig.to_csv(RESULTS_DIR / "significance_holm.csv", index=False)
    show = sig[["A", "B", "protocol", "runs", "delta_acc_pp", "p_paired_t", "p_wilcoxon",
                "p_t_holm_within_protocol", "p_t_holm_all"]]
    out += ["## 1. Multiple comparisons (Holm-Bonferroni on the run-level paired t-test)", "",
            f"{len(sig)} comparisons in `significance.csv`. With n = 6 runs the smallest possible two-sided "
            "Wilcoxon p is 0.031, so the Wilcoxon test cannot survive any correction.", "",
            show.round(4).to_markdown(index=False), ""]

    # 2) per-direction analysis of the SI comparisons
    pairs = [(("speech", "logmel"), ("speech", "wavlm")),
             (("speech", "wavlm"), ("fusion", "gated_wavlm_gt")),
             (("speech", "wavlm"), ("fusion", "gated_wavlm_asr")),
             (("speech", "wavlm"), ("fusion", "concat_wavlm_gt")),
             (("speech", "wavlm"), ("fusion", "xattn_wavlm_gt")),
             (("speech", "wavlm"), ("speech", "wavlm_spknorm"))]
    rows = []
    for a, b in pairs:
        r = dict(A="/".join(a), B="/".join(b))
        for fold, name in ((0, "OAF->YAF"), (1, "YAF->OAF")):
            d = np.array(run_acc(*b, fold)) - np.array(run_acc(*a, fold))
            r[f"delta {name} (3 seeds)"] = f"{100 * d.mean():+.2f} [{100 * d.min():+.1f}, {100 * d.max():+.1f}]"
            r[f"_s{fold}"] = np.sign(d.mean())
        r["same sign both directions"] = "yes" if r.pop("_s0") == r.pop("_s1") else "no"
        rows.append(r)
    out += ["## 2. SI comparisons per transfer direction", "",
            "Mean difference B - A over the 3 seeds of each direction (pp), with [min, max] over seeds. "
            "The 6 SI runs are not 6 independent samples: seeds within a direction share the training "
            "speaker and the test speaker. A difference that changes sign between directions is not "
            "established by this corpus.", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    # 3) regression-to-the-mean control for the confidence-binned fusion gain
    res = []
    for fus in ("gated_wavlm_asr", "gated_wavlm_gt"):
        for fold in (0, 1):
            for s in range(3):
                ref = load("speech", "wavlm", f"si_fold{fold}_seed{s}")
                g_f = binned_gain(ref, load("fusion", fus, f"si_fold{fold}_seed{s}"))
                for s2 in range(3):
                    if s2 == s:
                        continue
                    g_c = binned_gain(ref, load("speech", "wavlm", f"si_fold{fold}_seed{s2}"))
                    for b in BIN_LABELS:
                        res.append(dict(fusion=fus, fold=fold, seed=s, ctrl_seed=s2, bin=b, n=g_f.loc[b, "n"],
                                        gain_fusion=100 * (g_f.loc[b, "other"] - g_f.loc[b, "ref"]),
                                        gain_control=100 * (g_c.loc[b, "other"] - g_c.loc[b, "ref"])))
    res = pd.DataFrame(res)
    res.to_csv(RESULTS_DIR / "fusion_gain_regression_to_mean_control.csv", index=False)
    # weight each (run, bin) by clip count; control averaged over its 2 other seeds
    summ = (res.assign(wf=res.gain_fusion * res.n, wc=res.gain_control * res.n)
            .groupby(["fusion", "bin"], sort=False, observed=True)
            .agg(n=("n", "sum"), wf=("wf", "sum"), wc=("wc", "sum")))
    summ["fusion - speech (pp)"] = summ.wf / summ.n
    summ["other speech seed - speech (pp)"] = summ.wc / summ.n
    summ["net fusion effect (pp)"] = summ["fusion - speech (pp)"] - summ["other speech seed - speech (pp)"]
    summ["n"] = (summ.n / 2).astype(int)  # each run is counted once per control seed
    summ = summ.drop(columns=["wf", "wc"])
    out += ["## 3. Regression-to-the-mean control for 'fusion helps when speech is uncertain'", "",
            "Clips are binned by the confidence of the speech model (seed s). `fusion - speech` is the gain "
            "of the fusion model of the same seed (the figure in the report). `other speech seed - speech` "
            "is the same computation with a *second speech-only model* (another seed, same direction) in "
            "place of fusion: any gain it shows is a pure selection artefact. Only the net effect can be "
            "attributed to fusion.", "", summ.round(2).to_markdown(), ""]

    (RESULTS_DIR / "robustness_checks.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
