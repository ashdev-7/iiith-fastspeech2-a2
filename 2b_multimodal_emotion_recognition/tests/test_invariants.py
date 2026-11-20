"""Invariant tests for the silent bugs that would corrupt every number without crashing.

Each test targets one failure mode:
  * labels      : a clip's label/speaker disagrees with its file (mislabelled data)
  * splits      : a word or speaker leaks between train/val/test (inflated results)
  * features    : feature rows are misaligned with the index (model trained on the wrong clip)
  * normalise   : speaker statistics use labels or the wrong clips
  * predictions : saved predictions disagree with the split, the labels or the reported metrics
  * tables      : the published tables disagree with the per-run files

    python -m pytest tests/ -q                 (or: python tests/test_invariants.py)
Tests needing the raw data or the feature cache are skipped when those are absent (fresh clone).
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import CACHE_DIR, EMOTIONS, LABEL2ID, PROTOCOLS, RESULTS_DIR, RUNS_DIR, SPEAKERS  # noqa: E402
from common.data import load_index, make_split  # noqa: E402

DF = load_index()
HAVE_RAW = Path(DF.path.iloc[0]).exists()
HAVE_CACHE = (CACHE_DIR / "wavlm.data.npy").exists()
ALL_RUNS = [(p, s, f) for p, spec in PROTOCOLS.items() for s, f in spec["runs"]]


def skip(reason):
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        raise SkipTest(reason)


class SkipTest(Exception):
    pass


# ------------------------------------------------------------------ labels
def test_index_shape_and_balance():
    assert len(DF) == 2800 and DF.uid.is_unique
    assert (DF.groupby(["speaker", "emotion"]).size() == 200).all()
    assert (DF.groupby("word").size() == 14).all()
    assert (DF.label == DF.emotion.map(LABEL2ID)).all()


def test_label_and_speaker_match_the_file():
    fold2emo = {"pleasant_surprise": "ps", "pleasant_surprised": "ps"}
    for r in DF.itertuples():
        folder = Path(r.path).parent.name                      # e.g. OAF_angry
        spk, emo = folder.split("_", 1)
        assert spk == r.speaker, r.uid
        assert fold2emo.get(emo.lower(), emo.lower()) == r.emotion, r.uid
        stem = Path(r.path).stem.lower()                       # e.g. oaf_back_angry
        assert stem.split("_")[1] == r.word, r.uid
        assert stem.endswith(r.emotion if r.emotion != "ps" else "ps"), r.uid
        assert r.transcript == f"say the word {r.word}", r.uid


# ------------------------------------------------------------------ splits
def test_splits_are_disjoint_and_leak_free():
    for proto, seed, fold in ALL_RUNS:
        s = make_split(DF, proto, seed, fold)
        tr, va, te = (set(s[k]) for k in ("train", "val", "test"))
        assert not (tr & va) and not (tr & te) and not (va & te), (proto, seed, fold)
        w = {k: set(DF.word.values[list(v)]) for k, v in (("tr", tr), ("va", va), ("te", te))}
        spk = {k: set(DF.speaker.values[list(v)]) for k, v in (("tr", tr), ("va", va), ("te", te))}
        if proto == "sd":
            assert not (w["tr"] & w["te"]) and not (w["tr"] & w["va"]) and not (w["va"] & w["te"])
            assert len(tr | va | te) == 2800 and spk["tr"] == spk["te"] == set(SPEAKERS)
            assert (len(tr), len(va), len(te)) == (1960, 420, 420)
        elif proto == "si":
            test_spk = SPEAKERS[1 - fold]
            assert spk["te"] == {test_spk} and len(te) == 1400
            assert spk["tr"] == spk["va"] == {SPEAKERS[fold]}, "test speaker leaked into training/validation"
            assert not (w["tr"] & w["va"]) and (len(tr), len(va)) == (1190, 210)
        elif proto == "rand":
            assert len(tr | va | te) == 2800


def test_split_is_deterministic():
    for proto, seed, fold in ALL_RUNS:
        a, b = make_split(DF, proto, seed, fold), make_split(DF, proto, seed, fold)
        assert all((a[k] == b[k]).all() for k in a)


# ------------------------------------------------------------------ features
def test_feature_rows_align_with_index():
    if not HAVE_CACHE or not (CACHE_DIR / "waves_16k_trimmed.pkl").exists():
        skip("feature cache not available")
    from common.features import VarLenStore, _pooled_len, ssl_num_frames
    waves = pickle.loads((CACHE_DIR / "waves_16k_trimmed.pkl").read_bytes())
    s = VarLenStore("wavlm")
    assert len(waves) == len(s) == len(DF)
    expected = np.array([_pooled_len(ssl_num_frames(len(w))) for w in waves])
    got = np.array([s.length(i) for i in range(len(s))])
    # all 2800 frame counts must match the audio they were computed from, row by row
    assert (expected == got).all(), np.flatnonzero(expected != got)[:10]


def test_waves_match_raw_audio_durations():
    if not (HAVE_RAW and (CACHE_DIR / "waves_16k_trimmed.pkl").exists()):
        skip("raw audio not available")
    import warnings

    from scipy.io import wavfile
    waves = pickle.loads((CACHE_DIR / "waves_16k_trimmed.pkl").read_bytes())
    for r, w in zip(DF.itertuples(), waves):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sr, x = wavfile.read(r.path)
        except ValueError:          # the two AIFF files: checked separately in the report
            continue
        raw = len(x) / sr
        assert 0 < len(w) / 16000 <= raw + 1e-3, r.uid       # trimming only shortens the clip
        assert raw - len(w) / 16000 < 1.0, r.uid             # and never by a whole second


# ------------------------------------------------------------------ normalisation
def test_speaker_stats_are_label_free_and_correct():
    if not HAVE_CACHE:
        skip("feature cache not available")
    import inspect

    from common.features import VarLenStore, speaker_stats
    assert list(inspect.signature(speaker_stats).parameters) == ["name", "speakers", "cache_dir"]
    stats = speaker_stats("wavlm", DF.speaker.values)
    s = VarLenStore("wavlm")
    for spk in SPEAKERS:   # recompute from a fresh pass over that speaker's frames only
        rows = np.flatnonzero(DF.speaker.values == spk)
        x = np.concatenate([s[int(i)][:, 3].astype(np.float64) for i in rows])     # layer index 3 = L6
        assert np.allclose(stats[spk][0][3], x.mean(0), atol=1e-3)
        assert np.allclose(stats[spk][1][3], x.std(0), atol=1e-3)


# ------------------------------------------------------------------ predictions and metrics
def _run_files():
    for f in sorted(RUNS_DIR.glob("*/*/*/test_predictions.csv")):
        m = re.fullmatch(r"(sd|si|rand)_fold(\d+)_seed(\d+)", f.parent.name)
        yield f, m.group(1), int(m.group(3)), int(m.group(2))


def test_predictions_match_split_labels_and_metrics():
    n = 0
    for f, proto, seed, fold in _run_files():
        t = pd.read_csv(f)
        te = make_split(DF, proto, seed, fold)["test"]
        assert set(t.uid) == set(DF.uid.values[te]), f"{f}: predicted clips != test split"
        truth = DF.set_index("uid").loc[t.uid, "emotion"].values
        assert (t.true.values == truth).all(), f"{f}: wrong ground truth"
        p = t[[f"p_{e}" for e in EMOTIONS]].values
        assert np.allclose(p.sum(1), 1, atol=1e-3)
        argmax = np.array(EMOTIONS)[p.argmax(1)]
        assert (argmax == t.pred.values).mean() > 0.999, f"{f}: pred != argmax(p)"
        m = json.loads((f.parent / "test_metrics.json").read_text())
        assert abs(m["accuracy"] - (t.pred == t.true).mean()) < 1e-9, f"{f}: metric != predictions"
        n += 1
    assert n >= 224, n


def test_published_accuracy_table_recomputes():
    from common.results import collect
    runs = collect()
    names = {"Speech: WavLM + BiGRU": ("speech", "wavlm"), "Speech: log-Mel + BiGRU (baseline)": ("speech", "logmel"),
             "Fusion: gated GMU (GT)": ("fusion", "gated_wavlm_gt"), "Fusion: cross-attention (GT)": ("fusion", "xattn_wavlm_gt"),
             "Text: DistilRoBERTa + BiGRU (Whisper transcript)": ("text", "asr")}
    tab = pd.read_csv(RESULTS_DIR / "accuracy_table.csv").set_index("Model")
    for label, (pipe, var) in names.items():
        for proto in ("RAND", "SD", "SI"):
            r = runs[(runs.pipeline == pipe) & (runs.variant == var) & (runs.protocol == proto.lower())]
            r = r[[(proto.lower(), s, f) in ALL_RUNS for s, f in zip(r.seed, r.fold)]]
            mean, sd = 100 * r.accuracy.mean(), 100 * r.accuracy.std(ddof=1)
            pub_mean, pub_sd = (float(x) for x in tab.loc[label, f"{proto} Acc"].split(" ± "))
            assert abs(mean - pub_mean) < 0.006 and abs(sd - pub_sd) < 0.006, (label, proto, mean, pub_mean)


def test_asr_transcripts_align_with_index():
    from common.config import cached
    asr = pd.read_csv(cached("asr_transcripts.csv"))
    assert (asr.uid.values == DF.uid.values).all()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except SkipTest as e:
                print(f"SKIP  {name}: {e}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
