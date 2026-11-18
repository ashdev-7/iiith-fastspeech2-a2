"""Dataset indexing, transcripts and evaluation splits for TESS.

TESS facts that drive the design (verified in `build_index`):
  * 2 speakers (OAF, YAF) x 7 emotions x 200 target words = 2800 clips.
  * Every clip is the carrier phrase "Say the word <WORD>".
  * Every word is recorded in all 7 emotions by both speakers, so the lexical
    content is (by construction) statistically independent of the label.
  * The Kaggle archive contains a byte-identical nested copy of the corpus,
    which we ignore, and one oddly named file (`OA_bite_neutral.wav`), so
    speaker/emotion are read from the folder name, not the file name.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR, DATA_DIR, EMOTIONS, LABEL2ID, SPEAKERS

FOLDER_EMOTION = {
    "angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy",
    "neutral": "neutral", "pleasant_surprise": "ps", "pleasant_surprised": "ps", "sad": "sad",
}


def _find_corpus_root(data_dir: Path) -> Path:
    """Return the top-level folder holding the 14 `<SPK>_<emotion>` directories."""
    candidates = [p for p in data_dir.rglob("*") if p.is_dir() and re.fullmatch(r"(OAF|YAF)_\w+", p.name)]
    if not candidates:
        raise FileNotFoundError(f"No TESS speaker/emotion folders under {data_dir}. "
                                "Run `python prepare_data.py --download` first.")
    # shallowest parent wins (the archive nests an identical copy one level deeper)
    parents = sorted({c.parent for c in candidates}, key=lambda p: len(p.parts))
    return parents[0]


def clean_transcript(text: str) -> str:
    """Text preprocessing: lower-case, drop symbols, squeeze spaces.

    Sentence punctuation (. , ? !) is kept and split off as its own token: for
    ASR transcripts it is the only trace of prosody that survives in text.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9'.,?! ]+", " ", text)
    text = re.sub(r"\s*([.,?!])", r" \1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_index(data_dir: Path = DATA_DIR, save: bool = True) -> pd.DataFrame:
    root = _find_corpus_root(Path(data_dir))
    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not re.fullmatch(r"(OAF|YAF)_\w+", folder.name):
            continue
        speaker, emo_raw = folder.name.split("_", 1)
        emotion = FOLDER_EMOTION[emo_raw.lower()]
        for wav in sorted(folder.glob("*.wav")):
            parts = wav.stem.split("_")
            word = parts[1].lower()
            suffix = parts[-1].lower()
            assert suffix == emotion or (suffix == "ps" and emotion == "ps"), wav
            rows.append(dict(
                uid=f"{speaker}_{word}_{emotion}", path=str(wav), speaker=speaker,
                word=word, emotion=emotion, label=LABEL2ID[emotion],
                transcript=clean_transcript(f"Say the word {word}"),
            ))
    df = pd.DataFrame(rows).sort_values("uid").reset_index(drop=True)
    # integrity checks
    assert len(df) == 2800 and df.uid.is_unique, len(df)
    assert set(df.speaker) == set(SPEAKERS) and set(df.emotion) == set(EMOTIONS)
    assert (df.groupby("word").size() == 14).all(), "every word must appear 2 spk x 7 emo"
    if save:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_DIR / "index.csv", index=False)
    return df


def load_index() -> pd.DataFrame:
    path = CACHE_DIR / "index.csv"
    if not path.exists():  # fresh clone without the dataset: use the index exported from the Kaggle run
        from .config import ARTIFACTS_DIR
        if (ARTIFACTS_DIR / "index.csv").exists() and not DATA_DIR.exists():
            return pd.read_csv(ARTIFACTS_DIR / "index.csv")
        return build_index()
    return pd.read_csv(path)


def make_split(df: pd.DataFrame, protocol: str, seed: int, fold: int = 0) -> dict[str, np.ndarray]:
    """Return integer row indices for train / val / test.

    All splits are *word-disjoint* wherever the protocol allows it, so a model
    can never exploit the identity of a recording session or target word.
    """
    rng = np.random.default_rng(1000 + seed)
    words = np.array(sorted(df.word.unique()))
    rng.shuffle(words)
    if protocol == "sd":
        n_tr, n_va = 140, 30
        w_tr, w_va, w_te = words[:n_tr], words[n_tr:n_tr + n_va], words[n_tr + n_va:]
        tr = df.word.isin(w_tr)
        va = df.word.isin(w_va)
        te = df.word.isin(w_te)
    elif protocol == "si":
        train_spk = SPEAKERS[fold]
        test_spk = SPEAKERS[1 - fold]
        w_va = set(words[:30])
        tr = (df.speaker == train_spk) & ~df.word.isin(w_va)
        va = (df.speaker == train_spk) & df.word.isin(w_va)
        te = df.speaker == test_spk
    elif protocol == "rand":
        # the clip-level random split used by most TESS papers/notebooks (stratified by
        # speaker x emotion, 70/15/15). Same speakers AND same words in train and test:
        # reported only to show how optimistic this common protocol is.
        part = np.empty(len(df), dtype=object)
        for _, idx in df.groupby(["speaker", "emotion"]).indices.items():
            idx = rng.permutation(idx)
            n_tr, n_va = int(0.7 * len(idx)), int(0.15 * len(idx))
            part[idx[:n_tr]], part[idx[n_tr:n_tr + n_va]], part[idx[n_tr + n_va:]] = "train", "val", "test"
        tr, va, te = (pd.Series(part == k) for k in ("train", "val", "test"))
    else:
        raise ValueError(protocol)
    split = {k: np.flatnonzero(m.values) for k, m in (("train", tr), ("val", va), ("test", te))}
    assert not (set(split["train"]) & set(split["test"])) and not (set(split["train"]) & set(split["val"]))
    return split


def run_name(protocol: str, seed: int, fold: int) -> str:
    return f"{protocol}_fold{fold}_seed{seed}"
