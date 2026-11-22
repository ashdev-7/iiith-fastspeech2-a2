"""Build a blind annotation kit for a small hand-labelled reference .

Targets: all 15 words of the 3 ISLE prompts plus 45 words drawn at random (fixed seed) from the news
speech, i.e. 60 words and 120 boundaries (first planned as 12 news words; enlarged to 45 before any
marking was done, to narrow the confidence intervals). Random drawing means nobody chose the "easy" words.
For each target the kit gives only the word and its location rounded DOWN to the whole second
(so the annotator need not search 30 s of audio), never an aligner boundary.
The empty TextGrids contain a single tier "reference"; no aligner output is visible while annotating.

    python scripts/make_label_kit.py          -> hand_labels/kit/  (see hand_labels/ANNOTATION_PROTOCOL.md)
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_alignments import read_textgrid  # noqa: E402

KIT = ROOT / "hand_labels" / "kit"
N_NEWS = 45
SEED = 2026


def empty_textgrid(dur: float) -> str:
    return ('File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
            f"xmin = 0\nxmax = {dur}\ntiers? <exists>\nsize = 1\nitem []:\n"
            '    item [1]:\n        class = "IntervalTier"\n        name = "reference"\n'
            f"        xmin = 0\n        xmax = {dur}\n        intervals: size = 1\n"
            f'        intervals [1]:\n            xmin = 0\n            xmax = {dur}\n            text = ""\n')


def main():
    KIT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tg in sorted((ROOT / "outputs" / "arpa_base").glob("*/*.TextGrid")):
        ws = [(a, b, t) for a, b, t in read_textgrid(tg)["words"] if t]
        for k, (a, b, t) in enumerate(ws):
            rows.append(dict(file=tg.stem, speaker=tg.parent.name, word_index=k, word=t, _start=a))
    df = pd.DataFrame(rows)
    isle = df[df.file.str.startswith("ISLE")]
    news = df[~df.file.str.startswith("ISLE")].sample(N_NEWS, random_state=SEED)
    tgt = pd.concat([isle, news]).sort_values(["file", "word_index"])
    tgt["search_from_s"] = np.floor(tgt._start).astype(int)
    tgt.drop(columns="_start").to_csv(ROOT / "hand_labels" / "targets.csv", index=False)
    with open(KIT / "words_to_mark.txt", "w", encoding="utf-8") as f:
        for file, g in tgt.groupby("file", sort=False):
            f.write(f"{file}.wav\n")
            for r in g.itertuples():
                f.write(f"    word #{r.word_index:<3d} {r.word!r:<16} starts somewhere after {r.search_from_s} s\n")
            f.write("\n")
    for file in tgt.file.unique():
        spk = tgt[tgt.file == file].speaker.iloc[0]
        with wave.open(str(ROOT / "corpus" / spk / f"{file}.wav")) as w:
            dur = w.getnframes() / w.getframerate()
        for p in ("pass1", "pass2"):
            (KIT / p).mkdir(exist_ok=True)
            (KIT / p / f"{file}.TextGrid").write_text(empty_textgrid(dur), encoding="utf-8")
    print(f"{len(tgt)} target words ({2 * len(tgt)} boundaries) in {tgt.file.nunique()} files -> {KIT}")


if __name__ == "__main__":
    main()
