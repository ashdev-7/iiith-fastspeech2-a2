"""Score every alignment against the hand-labelled reference made with hand_labels/kit (two blind passes).

1. Annotator consistency: |pass1 - pass2| per boundary. This is the precision of the reference itself;
   aligner differences smaller than it cannot be resolved.
2. Aligner accuracy: |aligner - reference| where reference = mean of the two passes, per experiment, with the
   share of boundaries within 20 ms (the usual forced-alignment tolerance) and 50 ms.
3. Paired comparison of every experiment against arpa_base on the same boundaries (Wilcoxon).

    python scripts/score_hand_labels.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_alignments import read_textgrid  # noqa: E402

KIT = Path(os.environ.get("LABEL_KIT", ROOT / "hand_labels" / "kit"))
EXPERIMENTS = ["arpa_base", "arpa_adapted", "arpa_g2p_custom", "mfa_ipa", "mfa_ipa_dict_v300"]


def norm(w: str) -> str:
    return w.lower().strip().strip(".,'\"")


def read_pass(p: str, tgt: pd.DataFrame) -> pd.DataFrame:
    out = []
    for file, g in tgt.groupby("file", sort=False):
        ivs = [(a, b, norm(t)) for a, b, t in read_textgrid(KIT / p / f"{file}.TextGrid")["reference"] if t.strip()]
        if len(ivs) != len(g):
            raise SystemExit(f"{p}/{file}: {len(ivs)} labelled intervals, expected {len(g)} "
                             f"({', '.join(g.word)})")
        for (a, b, t), r in zip(ivs, g.itertuples()):
            if t != norm(r.word):
                raise SystemExit(f"{p}/{file}: interval labelled {t!r} where {r.word!r} was expected (in order)")
            out.append(dict(file=file, word_index=r.word_index, word=r.word, start=a, end=b))
    return pd.DataFrame(out)


def main():
    tgt = pd.read_csv(ROOT / "hand_labels" / "targets.csv")
    p1, p2 = read_pass("pass1", tgt), read_pass("pass2", tgt)
    g = p1.merge(p2, on=["file", "word_index", "word"], suffixes=("_1", "_2"))
    long = []
    for edge in ("start", "end"):
        d = g[["file", "word_index", "word"]].copy()
        d["edge"] = edge
        d["pass_diff_ms"] = 1000 * (g[f"{edge}_2"] - g[f"{edge}_1"])
        d["reference"] = (g[f"{edge}_1"] + g[f"{edge}_2"]) / 2
        long.append(d)
    ref = pd.concat(long, ignore_index=True)
    ref["speaker"] = np.where(ref.file.str.startswith("ISLE"), "isle", "f2b")

    errs = []
    for exp in EXPERIMENTS:
        for file, sub in ref.groupby("file"):
            spk = tgt[tgt.file == file].speaker.iloc[0]
            ws = [(a, b) for a, b, t in read_textgrid(ROOT / "outputs" / exp / spk / f"{file}.TextGrid")["words"] if t]
            for r in sub.itertuples():
                a, b = ws[r.word_index]
                errs.append(dict(experiment=exp, file=file, word_index=r.word_index, word=r.word, edge=r.edge,
                                 speaker=r.speaker,
                                 error_ms=1000 * ((a if r.edge == "start" else b) - r.reference)))
    e = pd.DataFrame(errs)
    e["abs"] = e.error_ms.abs()
    e.to_csv(ROOT / "analysis" / "hand_label_errors.csv", index=False)

    cons = ref.pass_diff_ms.abs()
    acc = e.groupby(["experiment", "speaker"], sort=False).agg(
        boundaries=("abs", "size"), median_abs_ms=("abs", "median"), mean_abs_ms=("abs", "mean"),
        within_20ms=("abs", lambda x: 100 * (x <= 20.0001).mean()),
        within_50ms=("abs", lambda x: 100 * (x <= 50.0001).mean()), max_abs_ms=("abs", "max")).round(1)
    allspk = e.groupby("experiment", sort=False).agg(
        boundaries=("abs", "size"), median_abs_ms=("abs", "median"), mean_abs_ms=("abs", "mean"),
        within_20ms=("abs", lambda x: 100 * (x <= 20.0001).mean()),
        within_50ms=("abs", lambda x: 100 * (x <= 50.0001).mean()), max_abs_ms=("abs", "max")).round(1)
    base = e[e.experiment == "arpa_base"].set_index(["file", "word_index", "edge"])["abs"]
    paired = []
    for exp in EXPERIMENTS[1:]:
        o = e[e.experiment == exp].set_index(["file", "word_index", "edge"])["abs"]
        d = (o - base).dropna()
        p = wilcoxon(d[d != 0]).pvalue if (d != 0).sum() >= 6 else np.nan
        paired.append(dict(experiment=exp, mean_extra_error_ms=round(d.mean(), 1),
                           boundaries_worse=int((d > 0).sum()), boundaries_better=int((d < 0).sum()),
                           p_wilcoxon=p))
    txt = ["# Hand-labelled reference (two blind passes)", "",
           f"{len(g)} words, {len(ref)} boundaries (`hand_labels/targets.csv`).", "",
           "## Annotator consistency (precision of the reference)", "",
           f"|pass1 - pass2|: median {cons.median():.1f} ms, mean {cons.mean():.1f} ms, "
           f"{100 * (cons <= 20.0001).mean():.0f} % within 20 ms, max {cons.max():.0f} ms.", "",
           "## Aligner error against the reference (reference = mean of the two passes)", "",
           allspk.to_markdown(), "", acc.to_markdown(), "",
           "## Paired against arpa_base (same boundaries)", "", pd.DataFrame(paired).to_markdown(index=False), ""]
    (ROOT / "analysis" / "hand_label_check.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
