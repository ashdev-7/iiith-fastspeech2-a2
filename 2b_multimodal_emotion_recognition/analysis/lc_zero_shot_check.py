"""Pre-registration 4, leakage control: what does emotion2vec+ seed's own head say on TESS? (POST HOC, descriptive)

The job scored LC-a at exactly 85.71 % (= 6/7) in both directions. This check shows why: 8 random clips per
speaker x emotion (112 clips, pandas seed 0), the head's unrestricted top label (9 classes, incl. "other").

    pip install funasr==1.4.28
    python analysis/lc_zero_shot_check.py   -> Results/lc_zero_shot_check.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.config import RESULTS_DIR  # noqa: E402
from common.data import load_index  # noqa: E402


def main():
    from funasr import AutoModel
    from prepare_data import load_waves
    df = load_index()
    waves = load_waves(df)
    m = AutoModel(model="emotion2vec/emotion2vec_plus_seed", hub="hf", disable_update=True)
    idx = df.groupby(["speaker", "emotion"]).sample(8, random_state=0).index
    rows = []
    for i in idx:
        r = m.generate(waves[i].astype(np.float32), granularity="utterance", extract_embedding=False,
                       disable_pbar=True)[0]
        rows.append(dict(speaker=df.speaker[i], emotion=df.emotion[i],
                         top=r["labels"][int(np.argmax(r["scores"]))].split("/")[-1]))
    r = pd.DataFrame(rows)
    tab = pd.crosstab([r.speaker, r.emotion], r.top)
    text = ("# Leakage control: emotion2vec+ seed's own head on TESS (post hoc, descriptive)\n\n"
            "8 random clips per speaker x emotion (112), no training on our side; unrestricted top label.\n\n"
            + tab.to_markdown() + "\n")
    (RESULTS_DIR / "lc_zero_shot_check.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
