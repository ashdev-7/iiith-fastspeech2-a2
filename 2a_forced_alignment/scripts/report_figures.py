"""Figures of report/report.pdf drawn at their printed size (text width 6.69 in: A4, 2 cm margins), as vector PDF.

Uses the same alignments and drawing functions as scripts/analyze_alignments.py; it computes no new result.
The Praat figures are drawn by scripts/praat/run_praat.py.

    python scripts/report_figures.py      -> analysis/figures/fig_overlay.pdf, fig_phone_durations.pdf (+ .png)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze_alignments import (CORPUS, EXP_LABEL, EXP_ORDER, FIG, SHORT_PHONE, SIL, load_experiments,  # noqa: E402
                                read_wav, spectrogram)

TEXT_WIDTH = 6.69
COLORS = ["#1f4e79", "#c00000", "#548235", "#7030a0"]


def style():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.dpi": 300})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)


def tier(ax, ivs, t0, t1, name, color):
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(name, rotation=0, ha="right", va="center")
    for s, e, lab in ivs:
        if e < t0 or s > t1:
            continue
        ax.axvline(s, color=color, lw=0.8)
        ax.axvline(e, color=color, lw=0.8)
        if lab and lab.lower() not in SIL:
            ax.text((max(s, t0) + min(e, t1)) / 2, 0.5, lab, ha="center", va="center", fontsize=8, clip_on=True)


def fig_overlay(exps, f="F2BJRLP2", t0=6.0, t1=10.0):
    """Word boundaries of the four experiments over one spectrogram (news speech, 6-10 s)."""
    x, sr = read_wav(next(CORPUS.rglob(f"{f}.wav")))
    names = [e for e in EXP_ORDER if e in exps]
    fig, axes = plt.subplots(1 + len(names), 1, figsize=(TEXT_WIDTH, 1.25 + 0.3 * len(names)), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2] + [0.62] * len(names), hspace=0.06))
    spectrogram(axes[0], x, sr, t0, t1)
    axes[0].set_ylabel("Hz")
    for ax, exp, c in zip(axes[1:], names, COLORS):
        d = exps[exp]
        ivs = list(zip(*[d[(d.file == f) & (d.tier == "words")][k] for k in ("start", "end", "label")]))
        tier(ax, ivs, t0, t1, EXP_LABEL[exp], c)
        for s, e, lab in ivs:
            if lab and lab.lower() not in SIL and t0 <= s <= t1:
                axes[0].axvline(s, color=c, lw=0.7, alpha=0.8)
    axes[-1].set_xlabel("time (s)")
    save(fig, "fig_overlay")
    plt.close(fig)


def fig_phone_durations(exps):
    fig, ax = plt.subplots(figsize=(4.4, 1.95))
    bins = np.arange(0, 0.31, 0.01)
    for exp, c in zip([e for e in EXP_ORDER if e in exps], COLORS):
        d = exps[exp]
        ax.hist(d[(d.tier == "phones") & ~d.is_sil].dur.clip(upper=0.3), bins=bins, histtype="step", lw=1.3,
                color=c, label=EXP_LABEL[exp])
    ax.axvline(SHORT_PHONE, color="k", ls=":", lw=0.9)
    ax.text(SHORT_PHONE - 0.004, ax.get_ylim()[1] * 0.93, "30 ms", fontsize=8, ha="right")
    ax.set_xlabel("phone duration (s, clipped at 0.3 s)")
    ax.set_ylabel("count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    save(fig, "fig_phone_durations")
    plt.close(fig)


def main():
    style()
    exps = load_experiments()
    fig_overlay(exps)
    fig_phone_durations(exps)
    print("wrote analysis/figures/fig_overlay.pdf, fig_phone_durations.pdf")


if __name__ == "__main__":
    main()
