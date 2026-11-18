"""One-off data preparation: download -> index -> preprocess -> cache features.

    python prepare_data.py --all          # everything below, in order
    python prepare_data.py --download     # Kaggle TESS archive into $MER_DATA_DIR
    python prepare_data.py --speech       # preprocess audio, WavLM + log-Mel features
    python prepare_data.py --text         # DistilRoBERTa features for ground-truth transcripts
    python prepare_data.py --asr          # Whisper transcripts + their text features (extension)
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from common.config import ASR_MODEL, CACHE_DIR, DATA_DIR, SAMPLE_RATE
from common.data import build_index, clean_transcript, load_index


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "kaggle", "datasets", "download",
                           "ejlok1/toronto-emotional-speech-set-tess", "-p", str(DATA_DIR), "--unzip"])


def load_waves(df: pd.DataFrame) -> list[np.ndarray]:
    path = CACHE_DIR / "waves_16k_trimmed.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    from common.audio import load_and_preprocess
    waves = [load_and_preprocess(p) for p in tqdm(df.path, desc="preprocess audio", mininterval=30)]
    path.write_bytes(pickle.dumps(waves))
    dur = np.array([len(w) / SAMPLE_RATE for w in waves])
    print(f"trimmed durations: mean {dur.mean():.2f}s  min {dur.min():.2f}s  max {dur.max():.2f}s")
    return waves


@torch.no_grad()
def transcribe(waves: list[np.ndarray], batch_size: int = 4) -> list[str]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    # MER_ASR_DEVICE=cpu forces CPU (used here: the laptop GPU thermally throttles to 300 MHz)
    device = os.environ.get("MER_ASR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    batch_size = 16 if device == "cuda" else batch_size  # small batches on CPU keep RAM < 1.5 GB
    proc = WhisperProcessor.from_pretrained(ASR_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL, torch_dtype=dtype).to(device).eval()
    out = []
    for i in tqdm(range(0, len(waves), batch_size), desc="Whisper", mininterval=10):
        feats = proc(waves[i:i + batch_size], sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        # greedy decoding, no prompt: punctuation/casing are Whisper's own, unbiased by us
        ids = model.generate(feats.to(device, dtype), language="en", task="transcribe", max_new_tokens=32)
        out += [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
    return out


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    for flag in ("all", "download", "speech", "speech_full", "text", "asr"):
        ap.add_argument(f"--{flag}", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    if args.download or args.all:
        download()
    df = build_index() if (args.all or args.download or not (CACHE_DIR / "index.csv").exists()) else load_index()
    print(df.groupby(["speaker", "emotion"]).size().unstack())

    if args.speech or args.all:
        from common.features import extract_logmel, extract_speech_ssl
        waves = load_waves(df)
        extract_logmel(waves)
        extract_speech_ssl(waves, device=DEVICE)

    if args.speech_full:  # ablation store: all 13 WavLM layers at native 20 ms (~5.6 GB, not part of --all)
        from common.features import SSL_FULL_LAYERS, extract_speech_ssl
        extract_speech_ssl(load_waves(df), name="wavlm_full", device=DEVICE, layers=SSL_FULL_LAYERS, pool=1)

    if args.text or args.all:
        from common.features import extract_text
        extract_text(df.transcript.tolist(), name="text_gt", device=DEVICE)

    if args.asr or args.all:
        from common.features import extract_text
        asr_csv = CACHE_DIR / "asr_transcripts.csv"
        if not asr_csv.exists():
            raw = transcribe(load_waves(df))
            pd.DataFrame({"uid": df.uid, "asr_raw": raw}).to_csv(asr_csv, index=False)
        asr = pd.read_csv(asr_csv).fillna("")
        assert (asr.uid.values == df.uid.values).all()
        extract_text([clean_transcript(t) for t in asr.asr_raw], name="text_asr", device=DEVICE)


if __name__ == "__main__":
    main()
