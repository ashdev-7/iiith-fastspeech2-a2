"""Pre-registered SI experiment: low-capacity classifier (see analysis/PREREGISTRATION_si_linear.md).

All-layer pooled WavLM statistics -> StandardScaler -> L2 logistic regression. C is chosen by 5-fold
cross-validation grouped by word on the TRAINING speaker only; the test speaker is used once, for the
final evaluation. Compared clip-by-clip with the 6 SI runs of WavLM + BiGRU (McNemar).

    python analysis/si_linear.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analysis.layer_probe import pooled, probe, spk_norm  # noqa: E402
from common.config import EMOTIONS, PROJECT_ROOT, RESULTS_DIR, SPEAKERS  # noqa: E402
from common.data import load_index, make_split  # noqa: E402

C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def clf(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def select_C(X, y, groups):
    cv = GroupKFold(n_splits=5)
    scores = {}
    for C in C_GRID:
        acc = [(clf(C).fit(X[a], y[a]).predict(X[b]) == y[b]).mean() for a, b in cv.split(X, y, groups)]
        scores[C] = float(np.mean(acc))
    return max(scores, key=scores.get), scores


def mcnemar(a_ok, b_ok):
    """Exact McNemar test on paired correctness vectors; returns (fixes, breaks, p)."""
    fixes, breaks = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = binomtest(fixes, fixes + breaks, 0.5).pvalue if fixes + breaks else 1.0
    return fixes, breaks, float(p)


def main():
    df = load_index()
    y, spk, words = df.label.values, df.speaker.values, df.word.values
    P = pooled("wavlm")                                    # (N, 7 layers, 1536)

    # sanity: this store must reproduce the per-layer probes in Results/layer_probe.csv
    ref = pd.read_csv(RESULTS_DIR / "layer_probe.csv").query("store == 'wavlm'")
    si = [make_split(df, "si", 0, f) for f in (0, 1)]
    trte = [(np.concatenate([s["train"], s["val"]]), s["test"]) for s in si]
    for li, row in enumerate(ref.itertuples()):
        got = np.mean([probe(P[:, li], y, a, b) for a, b in trte])
        assert abs(got - row.emotion_si) < 0.005, (row.layer, got, row.emotion_si)
    print("sanity: per-layer SI probes reproduce Results/layer_probe.csv", flush=True)

    X_all = P.reshape(len(P), -1)
    feats = {"all-layer linear": X_all, "all-layer linear + speaker norm †": spk_norm(X_all, spk)}
    preds = {"uid": df.uid.values}
    rows, sel = [], {}
    for name, X in feats.items():
        for fold, (tr, te) in enumerate(trte):
            C, cv = select_C(X[tr], y[tr], words[tr])
            p = clf(C).fit(X[tr], y[tr]).predict(X[te])
            preds.setdefault(name, np.full(len(df), -1))[te] = p
            sel[f"{name} | train {SPEAKERS[fold]}"] = {"C": C, "cv_acc_train_speaker": cv}
            rows.append(dict(model=name, train=SPEAKERS[fold], test=SPEAKERS[1 - fold], C=C,
                             cv_acc_train_speaker=round(100 * cv[C], 2),
                             accuracy=round(100 * (p == y[te]).mean(), 2),
                             macro_f1=round(100 * f1_score(y[te], p, average="macro"), 2)))
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows)

    # clip-by-clip comparison with the 6 SI runs of WavLM + BiGRU (and its speaker-norm variant)
    lab = {e: i for i, e in enumerate(EMOTIONS)}
    uid2row = {u: i for i, u in enumerate(df.uid)}
    comp = []
    for base, base_dir in (("WavLM + BiGRU", "wavlm"), ("WavLM + BiGRU + speaker norm †", "wavlm_spknorm")):
        for run in sorted((PROJECT_ROOT / "runs" / "speech" / base_dir).glob("si_fold*_seed*")):
            t = pd.read_csv(run / "test_predictions.csv")
            idx = t.uid.map(uid2row).values
            b_ok = t.pred.map(lab).values == y[idx]
            for name in feats:
                a_ok = preds[name][idx] == y[idx]
                f, br, p = mcnemar(a_ok, b_ok)
                comp.append(dict(model=name, baseline=base, run=run.name, acc_model=round(100 * a_ok.mean(), 2),
                                 acc_baseline=round(100 * b_ok.mean(), 2),
                                 delta_pp=round(100 * (a_ok.mean() - b_ok.mean()), 2),
                                 fixes=f, breaks=br, p_mcnemar=p))
    comp = pd.DataFrame(comp)

    res.to_csv(RESULTS_DIR / "si_linear.csv", index=False)
    comp.to_csv(RESULTS_DIR / "si_linear_vs_bigru.csv", index=False)
    pd.DataFrame(preds).to_csv(RESULTS_DIR / "si_linear_predictions.csv", index=False)
    (RESULTS_DIR / "si_linear_selection.json").write_text(json.dumps(sel, indent=2))

    summ = res.groupby("model", sort=False).agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"))
    cs = comp.groupby(["model", "baseline"], sort=False).agg(
        mean_delta_pp=("delta_pp", "mean"), runs_better=("delta_pp", lambda d: f"{(d > 0).sum()}/{len(d)}"),
        max_p_mcnemar=("p_mcnemar", "max"))
    (RESULTS_DIR / "si_linear.md").write_text(
        "# Pre-registered SI experiment: all-layer pooled WavLM + logistic regression\n\n"
        "Plan: `analysis/PREREGISTRATION_si_linear.md`. C chosen by word-grouped 5-fold CV on the training "
        "speaker only. † = transductive (label-free) speaker normalisation.\n\n"
        "## Per direction\n\n" + res.to_markdown(index=False) +
        "\n\n## Mean over the two directions\n\n" + summ.round(2).to_markdown() +
        "\n\n## Against the 6 SI runs of the BiGRU models (same test clips)\n\n" + cs.round(4).to_markdown() +
        "\n\nPer-run McNemar results: `si_linear_vs_bigru.csv`.\n", encoding="utf-8")
    print((RESULTS_DIR / "si_linear.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
