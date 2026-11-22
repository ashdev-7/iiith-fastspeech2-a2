"""Print style for the report figures.

Every report figure is drawn at the size it is printed (the report's text width is 6.77 in: A4 with 1.9 cm margins),
so 8-9 pt in the figure is 8-9 pt on paper, close to the body text. Figures are saved as vector PDF (for the report)
and PNG (for the README and quick viewing). TrueType fonts are embedded (pdf.fonttype 42), never Type 3.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

TEXT_WIDTH = 6.77           # inches
HALF_WIDTH = 3.3            # inches: one of two side-by-side figures


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def save(fig, stem: Path):
    """Write stem.pdf (vector, for the report) and stem.png (for viewing)."""
    stem = Path(stem)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
