"""Ablation of two speech front-end assumptions, paired against the main WavLM model.

  wavlm        : main model, 7 layers (0,2,...,12) at 40 ms, weight decay on layer logits
  wavlm_full   : all 13 layers at native 20 ms
  wavlm_lw0    : no weight decay on the layer-weight logits
  wavlm_lw0x10 : no weight decay + 10x learning rate on the layer-weight logits

Outputs: Results/ablation_speech.md, Results/ablation_speech.csv, Results/plots/ablation_layer_weights.png

    python analysis/ablation_speech.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from common.config import PLOTS_DIR, PROTOCOLS, RESULTS_DIR, RUNS_DIR  # noqa: E402

VARIANTS = {"wavlm": "main: 7 layers, 40 ms", "wavlm_full": "all 13 layers, 20 ms",
            "wavlm_lw0": "layer logits: no weight decay", "wavlm_lw0x10": "layer logits: no decay, 10x lr"}
LAYERS = {"wavlm": [0, 2, 4, 6, 8, 10, 12], "wavlm_full": list(range(13)),
          "wavlm_lw0": [0, 2, 4, 6, 8, 10, 12], "wavlm_lw0x10": [0, 2, 4, 6, 8, 10, 12]}


def runs(var: str, proto: str) -> dict[str, dict]:
    out = {}
    for f in sorted((RUNS_DIR / "speech" / var).glob(f"{proto}_*/test_metrics.json")):
        m = json.loads(f.read_text())
        out[f.parent.name] = dict(acc=100 * m["accuracy"], w=np.array(next(iter(m["layer_weights"].values()))))
    return out


def main():
    rows, wts = [], {}
    for var, desc in VARIANTS.items():
        for proto in PROTOCOLS:
            r = runs(var, proto)
            if not r:
                continue
            base = runs("wavlm", proto)
            acc = np.array([v["acc"] for v in r.values()])
            W = np.stack([v["w"] for v in r.values()])
            wts[(var, proto)] = W.mean(0)
            row = dict(variant=var, description=desc, protocol=proto, runs=len(acc),
                       acc_mean=acc.mean(), acc_std=acc.std(ddof=1) if len(acc) > 1 else np.nan,
                       max_layer_weight=W.max(1).mean(), uniform_weight=1 / W.shape[1],
                       top_layer=LAYERS[var][int(W.mean(0).argmax())])
            if var != "wavlm":
                common = sorted(set(r) & set(base))
                d = np.array([r[k]["acc"] - base[k]["acc"] for k in common])
                if len(d) > 1:
                    row.update(paired_runs=len(d), delta_pp=d.mean(),
                               ci95_pp=stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)),
                               p_paired_t=stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else np.nan)
            rows.append(row)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(RESULTS_DIR / "ablation_speech.csv", index=False)
    md = ["# Speech front-end ablation (test accuracy %, paired against the main WavLM model)", "",
          "delta = mean per-run difference vs `wavlm` on identical splits/seeds, with 95 % t-CI and paired t-test. "
          "max_layer_weight = mean over runs of the largest softmax layer weight (uniform = 1/L).", "",
          t.round(3).to_markdown(index=False), ""]
    (RESULTS_DIR / "ablation_speech.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(12, 3.2), sharey=True)
    for ax, proto in zip(np.atleast_1d(axes), PROTOCOLS):
        for var, c in zip(VARIANTS, ["#0072B2", "#7F7F7F", "#E69F00", "#D55E00"]):
            if (var, proto) in wts:
                ax.plot(LAYERS[var], wts[(var, proto)], "o-", color=c, label=VARIANTS[var], ms=4)
        ax.axhline(1 / 7, ls=":", color="k", lw=0.8)
        ax.axhline(1 / 13, ls=":", color="#7F7F7F", lw=0.8)
        ax.set_title(proto.upper(), fontsize=9)
        ax.set_xlabel("WavLM layer")
        ax.set_xticks(range(0, 13, 2))
    np.atleast_1d(axes)[0].set_ylabel("learned softmax weight (mean over runs)")
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.suptitle("Learned layer weights: main model vs. ablations (dotted: uniform 1/7 and 1/13)", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ablation_layer_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
