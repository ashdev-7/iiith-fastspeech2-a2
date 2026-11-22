"""Browser alternative to Praat for the blind hand-marking passes (hand_labels/ANNOTATION_PROTOCOL.md).

    python scripts/marking_tool.py build
        -> hand_labels/kit/marking_tool.html (self-contained: the six recordings and the target words are
           embedded; it shows no aligner output). Open it in a browser, mark, and it downloads
           passN_marks.json.
    python scripts/marking_tool.py convert <passN_marks.json> [...]
        -> hand_labels/kit/passN/<file>.TextGrid, the same files the Praat route produces, so
           scripts/score_hand_labels.py reads them unchanged. Boundaries are moved to the nearest zero crossing
           (protocol rule 7; only if one lies within 2.5 ms, otherwise the click is kept); a word marked "no pause" shares its start
           boundary with the previous word's end (rule 6).
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "hand_labels" / "kit"


def wav_path(file: str, speaker: str) -> Path:
    return ROOT / "corpus" / speaker / f"{file}.wav"


def build():
    tgt = pd.read_csv(ROOT / "hand_labels" / "targets.csv")
    audio, sentences, sr = {}, {}, None
    for (file, spk), _ in tgt.groupby(["file", "speaker"], sort=False):
        x, sr_f = sf.read(wav_path(file, spk), dtype="int16")
        assert x.ndim == 1 and (sr is None or sr == sr_f)
        sr = sr_f
        audio[file] = base64.b64encode(x.astype("<i2").tobytes()).decode()
        if file.startswith("ISLE"):
            sentences[file] = wav_path(file, spk).with_suffix(".lab").read_text(encoding="utf-8").strip().lower()
    data = dict(sample_rate=sr, audio=audio, sentences=sentences,
                targets=tgt[["file", "word_index", "word", "search_from_s"]].to_dict("records"))
    html = (KIT / "marking_tool_template.html").read_text(encoding="utf-8")
    assert html.count("/*DATA*/null") == 1
    out = KIT / "marking_tool.html"
    out.write_text(html.replace("/*DATA*/null", json.dumps(data)), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def nearest_zero_crossing(x: np.ndarray, sr: int, t: float) -> float:
    """Time of the zero crossing closest to t (linear interpolation between samples), as Praat does."""
    i = int(round(t * sr))
    lo, hi = max(1, i - sr // 400), min(len(x) - 1, i + sr // 400)   # search +-2.5 ms, else keep t
    seg = x[lo - 1:hi + 1].astype(float)
    k = np.nonzero(np.signbit(seg[:-1]) != np.signbit(seg[1:]))[0]
    if not len(k):
        return t
    a, b = seg[k], seg[k + 1]
    times = (lo - 1 + k + a / (a - b)) / sr
    return float(times[np.argmin(np.abs(times - t))])


def write_textgrid(path: Path, xmax: float, ivs: list[tuple[float, float, str]]):
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {xmax!r}",
             "tiers? <exists>", "size = 1", "item []:", "    item [1]:", '        class = "IntervalTier"',
             '        name = "reference"', "        xmin = 0", f"        xmax = {xmax!r}",
             f"        intervals: size = {len(ivs)}"]
    for n, (a, b, t) in enumerate(ivs, 1):
        lines += [f"        intervals [{n}]:", f"            xmin = {a!r}", f"            xmax = {b!r}",
                  f'            text = "{t}"']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert(path: str):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    p = int(d["pass"])
    assert p in (1, 2)
    tgt = pd.read_csv(ROOT / "hand_labels" / "targets.csv")
    m = pd.DataFrame(d["marks"])
    assert len(m) == len(tgt) and (m.file.values == tgt.file.values).all() \
        and (m.word_index.values == tgt.word_index.values).all(), "marks do not match hand_labels/targets.csv"
    assert m.start.notna().all() and m.end.notna().all(), "some words are not marked"
    for (file, spk), g in tgt.groupby(["file", "speaker"], sort=False):
        x, sr = sf.read(wav_path(file, spk), dtype="int16")
        xmax = len(x) / sr
        bounds, words, prev_end = [], [], None
        for r in m[m.file == file].itertuples():
            s = prev_end if (r.touches_prev and prev_end is not None) else nearest_zero_crossing(x, sr, r.start)
            e = nearest_zero_crossing(x, sr, r.end)
            assert e - s > 0.01 and (prev_end is None or s >= prev_end - 1e-9), (file, r.word, s, e)
            words.append((s, e, r.word))
            prev_end = e
        ivs, t = [], 0.0
        for s, e, w in words:
            if s > t + 1e-9:
                ivs.append((t, s, ""))
            ivs.append((s, e, w))
            t = e
        if t < xmax - 1e-9:
            ivs.append((t, xmax, ""))
        out = KIT / f"pass{p}" / f"{file}.TextGrid"
        write_textgrid(out, xmax, ivs)
        print(f"pass {p}: {out.name}: {len(words)} words")


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["convert"] and len(sys.argv) > 2:
        for f in sys.argv[2:]:
            convert(f)
    else:
        raise SystemExit(__doc__)
