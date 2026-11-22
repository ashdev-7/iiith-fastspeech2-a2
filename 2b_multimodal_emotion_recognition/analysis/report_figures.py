"""The figures of report/report.pdf, drawn at their printed size from saved results (common/plotstyle.py).

Reads only saved outputs (runs/, Results/*.csv and the cached waveforms) and writes Results/plots/fig_*.pdf and .png.
It recomputes no result: every number shown comes from the same files the analysis scripts wrote.

    python analysis/report_figures.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from common import plotstyle  # noqa: E402
from common.audio import log_mel  # noqa: E402
from common.config import CACHE_DIR, EMOTION_NAMES, EMOTIONS, PLOTS_DIR, RESULTS_DIR, RUNS_DIR, SAMPLE_RATE  # noqa: E402
from common.data import load_index  # noqa: E402

PALETTE = dict(zip(EMOTIONS, ["#D55E00", "#009E73", "#7F3C8D", "#E69F00", "#7F7F7F", "#56B4E9", "#0072B2"]))
SHORT = {e: EMOTION_NAMES[e].replace("Pleasant surprise", "Pl. surprise") for e in EMOTIONS}
OUT = PLOTS_DIR


def fig_layer_weights():
    """Learned softmax layer weights of the main model and the front-end ablations (Table: ablation)."""
    variants = {"wavlm": ("main: 7 layers, 40 ms", [0, 2, 4, 6, 8, 10, 12], "#0072B2"),
                "wavlm_full": ("all 13 layers, 20 ms", list(range(13)), "#7F7F7F"),
                "wavlm_lw0": ("weights: no weight decay", [0, 2, 4, 6, 8, 10, 12], "#E69F00"),
                "wavlm_lw0x10": ("weights: no decay, 10x learning rate", [0, 2, 4, 6, 8, 10, 12], "#D55E00")}
    names = {"rand": "Random split", "sd": "Seen speakers (SD)", "si": "Unseen speaker (SI)"}
    fig, axes = plt.subplots(1, 3, figsize=(plotstyle.TEXT_WIDTH, 2.3), sharey=True)
    for ax, proto in zip(axes, ("rand", "sd", "si")):
        for var, (lab, layers, c) in variants.items():
            W = [np.array(next(iter(json.loads(f.read_text())["layer_weights"].values())))
                 for f in sorted((RUNS_DIR / "speech" / var).glob(f"{proto}_*/test_metrics.json"))]
            if W:
                ax.plot(layers, np.mean(W, 0), "o-", color=c, label=lab, ms=3)
        ax.axhline(1 / 7, ls=":", color="k", lw=0.8)
        ax.axhline(1 / 13, ls=":", color="#7F7F7F", lw=0.8)
        ax.set_title(names[proto])
        ax.set_xlabel("WavLM layer")
        ax.set_xticks(range(0, 13, 2))
    axes[0].set_ylabel("learned weight")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plotstyle.save(fig, OUT / "fig_layer_weights")
    plt.close(fig)


def _confusion(ax, pipe, var, proto, title):
    mats = [np.array(json.loads(f.read_text())["confusion"])
            for f in (RUNS_DIR / pipe / var).glob(f"{proto}_*/test_metrics.json")]
    cm = np.sum(mats, 0)
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{100 * cmn[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if cmn[i, j] > 0.6 else "black")
    ticks = [SHORT[e] for e in EMOTIONS]
    ax.set_xticks(range(7), ticks, rotation=45, ha="right")
    ax.set_yticks(range(7), ticks)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title} ({len(mats)} runs, row %)")
    for s in ax.spines.values():
        s.set_visible(False)


def fig_confusions():
    fig, axes = plt.subplots(1, 2, figsize=(plotstyle.TEXT_WIDTH, 3.3))
    _confusion(axes[0], "speech", "wavlm", "si", "Speech-only, unseen speaker")
    _confusion(axes[1], "text", "asr", "sd", "Text-only (Whisper), seen speakers")
    fig.tight_layout(w_pad=2.5)
    plotstyle.save(fig, OUT / "fig_confusions")
    plt.close(fig)


def fig_fusion():
    fig, axes = plt.subplots(1, 2, figsize=(plotstyle.TEXT_WIDTH, 2.5))
    ax = axes[0]
    for var, c, lab in (("gated_wavlm_gt", "#0072B2", "ground-truth text"), ("gated_wavlm_asr", "#D55E00", "Whisper text")):
        gs = [pd.read_csv(f).gate_speech for f in sorted((RUNS_DIR / "fusion" / var).glob("*/test_predictions.csv"))]
        ax.hist(pd.concat(gs), bins=40, range=(0, 1), alpha=0.6, color=c, label=lab, density=True)
    ax.set_xlabel("mean gate $z$ per clip (1 = speech, 0 = text)")
    ax.set_ylabel("density")
    ax.set_title("Reliance on speech vs. text")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    g = pd.read_csv(RESULTS_DIR / "fusion_gain_by_speech_confidence_gated_wavlm_asr_si.csv")
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.speech_acc, 0.4, label="speech-only", color="#0072B2")
    ax.bar(x + 0.2, g.fusion_acc, 0.4, label="fusion (Whisper text)", color="#E69F00")
    names = {"[0.0, 0.5)": "<0.5", "[0.5, 0.7)": "0.5-0.7", "[0.7, 0.9)": "0.7-0.9", "[0.9, 0.99)": "≥0.9"}
    assert list(g.conf_bin) == list(names), "bins differ from the report table"   # no clip has confidence >= 0.99
    labels = [names[b] for b in g.conf_bin]
    ax.set_xticks(x, [f"{lab}\n$n$={int(n)}" for lab, n in zip(labels, g.n)])
    ax.set_xlabel("speech-only confidence (max probability)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Unseen speaker: accuracy by confidence")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2.0)
    plotstyle.save(fig, OUT / "fig_fusion")
    plt.close(fig)


def fig_error_cases():
    from analysis.error_analysis import MODEL, prosody
    df = load_index()
    waves = pickle.loads((CACHE_DIR / "waves_16k_trimmed.pkl").read_bytes())
    uid2i = {u: i for i, u in enumerate(df.uid)}
    P = pd.concat([pd.read_csv(f) for f in sorted((RUNS_DIR / MODEL[0] / MODEL[1]).glob("si_*/test_predictions.csv"))])
    pcols = [f"p_{e}" for e in EMOTIONS]
    probs = P.groupby("uid")[pcols].mean()
    true = P.groupby("uid").true.first()
    pred = P.groupby("uid").pred.agg(lambda s: s.value_counts().index[0])
    cases = pd.read_csv(RESULTS_DIR / "error_analysis_cases.csv")["clip"].tolist()
    fig, axes = plt.subplots(len(cases), 2, figsize=(plotstyle.TEXT_WIDTH, 1.75 * len(cases)),
                             gridspec_kw=dict(width_ratios=[3, 1.15]))
    for k, (ax_row, c) in enumerate(zip(axes, cases)):
        y = waves[uid2i[c]]
        pr = prosody(y)
        ax = ax_row[0]
        ax.imshow(log_mel(y).T, origin="lower", aspect="auto", cmap="magma", extent=(0, len(y) / SAMPLE_RATE, 0, 80))
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(pr["f0_track"])) * 160 / SAMPLE_RATE, pr["f0_track"], color="#56B4E9", lw=1.4)
        ax2.set_ylim(75, 600)
        ax2.set_ylabel("F0 (Hz)")
        ax2.spines["right"].set_visible(True)
        ax.set_ylabel("mel bin")
        ax.set_title(f"{c.replace('_', ' ')}: true {EMOTION_NAMES[true[c]].lower()}, predicted {EMOTION_NAMES[pred[c]].lower()}",
                     loc="left")
        if k == len(cases) - 1:
            ax.set_xlabel("time (s)")
        colors = ["#D55E00" if e == pred[c] else "#009E73" if e == true[c] else "#BBBBBB" for e in EMOTIONS]
        b = ax_row[1]
        b.barh(range(7), probs.loc[c].values, color=colors)
        b.set_yticks(range(7), [SHORT[e] for e in EMOTIONS])
        b.set_xlim(0, 1)
        b.invert_yaxis()
        if k == len(cases) - 1:
            b.set_xlabel("mean probability")
    fig.tight_layout(h_pad=0.6)
    plotstyle.save(fig, OUT / "fig_error_cases")
    plt.close(fig)


def fig_tsne():
    blocks = [("speech", "wavlm", "temporal", "Temporal block (speech)"),
              ("text", "gt", "contextual", "Contextual block (text)"),
              ("fusion", "gated_wavlm_gt", "fusion", "Fusion block")]
    protos = (("sd", "sd_fold0_seed0", "Seen speakers (SD)"), ("si", "si_fold0_seed0", "Unseen speaker (SI)"))
    fig, axes = plt.subplots(2, 3, figsize=(plotstyle.TEXT_WIDTH, 4.7))
    for r, (proto, run, pname) in enumerate(protos):
        for c, (pipe, var, key, title) in enumerate(blocks):
            ax = axes[r, c]
            z = np.load(RUNS_DIR / pipe / var / run / "test_embeddings.npz")
            X, y = z[key], z["label"]
            Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
            sil = silhouette_score(X, y, metric="cosine")
            for e_i, e in enumerate(EMOTIONS):
                m = y == e_i
                ax.scatter(Z[m, 0], Z[m, 1], s=4, c=PALETTE[e], alpha=0.85, linewidths=0, label=EMOTION_NAMES[e],
                           rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.5)
            ax.text(0.03, 0.03, f"silhouette {sil:.2f}", transform=ax.transAxes, fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="0.7", lw=0.5, pad=1.5))
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(pname)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=7, frameon=False, markerscale=2.5, handletextpad=0.2,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.5, w_pad=0.5)
    plotstyle.save(fig, OUT / "fig_tsne")
    plt.close(fig)


def fig_noise():
    from analysis.noise_robustness import MODELS, SNRS
    t = pd.read_csv(RESULTS_DIR / "noise_robustness_runs.csv", dtype={"snr": str})
    order = ["clean"] + [str(s) for s in SNRS]
    g = t.groupby(["protocol", "model", "snr"]).acc.agg(["mean", "std"]).reset_index()
    colors = ["#0072B2", "#56B4E9", "#7F7F7F", "#009E73", "#D55E00", "#E69F00"]
    fig, axes = plt.subplots(1, 2, figsize=(plotstyle.TEXT_WIDTH, 2.6), sharey=True)
    for ax, proto in zip(axes, ("sd", "si")):
        for (pipe, var, label, *_), c in zip(MODELS, colors):
            d = g[(g.protocol == proto) & (g.model == label)].set_index("snr").reindex(order)
            ax.errorbar(range(len(order)), 100 * d["mean"], yerr=100 * d["std"], marker="o", ms=3, capsize=2,
                        lw=1.1, label=label, color=c)
        ax.axhline(100 / 7, ls=":", color="k", lw=0.8)
        ax.set_xticks(range(len(order)), ["clean"] + [f"{s} dB" for s in SNRS])
        ax.set_title({"sd": "Seen speakers (SD)", "si": "Unseen speaker (SI)"}[proto])
        ax.set_xlabel("test-time signal-to-noise ratio")
    axes[0].set_ylabel("accuracy (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    plotstyle.save(fig, OUT / "fig_noise")
    plt.close(fig)


def main():
    plotstyle.apply()
    for f in (fig_layer_weights, fig_confusions, fig_fusion, fig_error_cases, fig_tsne, fig_noise):
        f()
        print(f.__name__, "done", flush=True)


if __name__ == "__main__":
    main()
