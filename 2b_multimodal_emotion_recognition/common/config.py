"""Global configuration shared by all three pipelines.

Paths can be overridden with environment variables so that the code runs
unchanged on another machine:
    MER_DATA_DIR   - folder that contains the extracted Kaggle TESS archive
    MER_CACHE_DIR  - folder for cached features (several GB, keep it off cloud-synced drives)
    MER_RUNS_DIR   - folder for checkpoints / per-run outputs
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("MER_DATA_DIR", Path.home() / "data" / "tess"))
CACHE_DIR = Path(os.environ.get("MER_CACHE_DIR", Path.home() / "data" / "tess_cache"))
RUNS_DIR = Path(os.environ.get("MER_RUNS_DIR", PROJECT_ROOT / "runs"))
RESULTS_DIR = PROJECT_ROOT / "Results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)  # analyses may run before make_results.py creates it

# ---------------------------------------------------------------- labels
# Order is fixed so that integer labels are identical across all scripts.
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
EMOTION_NAMES = {
    "angry": "Angry", "disgust": "Disgust", "fear": "Fear", "happy": "Happy",
    "neutral": "Neutral", "ps": "Pleasant surprise", "sad": "Sad",
}
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
NUM_CLASSES = len(EMOTIONS)
SPEAKERS = ["OAF", "YAF"]  # older (64 y) and younger (26 y) adult female

# ---------------------------------------------------------------- audio
SAMPLE_RATE = 16_000        # TESS is 24.414 kHz; 16 kHz is what WavLM was pre-trained on
TRIM_TOP_DB = 30            # leading/trailing silence threshold (dB below peak)
TRIM_PAD_S = 0.05           # keep 50 ms of context around the trimmed region
MAX_AUDIO_S = 4.0           # safety cap (longest TESS clip after trimming is ~2.5 s)

# ---------------------------------------------------------------- pretrained encoders (frozen)
SPEECH_SSL_MODEL = "microsoft/wavlm-base-plus"
TEXT_PLM_MODEL = "j-hartmann/emotion-english-distilroberta-base"
ASR_MODEL = "openai/whisper-small"

# The emotion-tuned DistilRoBERTa predicts 7 classes that map 1:1 onto TESS.
TEXT_PLM_LABEL_MAP = {
    "anger": "angry", "disgust": "disgust", "fear": "fear", "joy": "happy",
    "neutral": "neutral", "surprise": "ps", "sadness": "sad",
}

# ---------------------------------------------------------------- evaluation protocols
# sd : speaker-dependent, *word-disjoint* (both speakers in every split, a word's
#      14 recordings live in exactly one split). 5 seeds.
# si : speaker-independent (leave-one-speaker-out). Train/val on one speaker
#      (word-disjoint), test on all 1400 clips of the unseen speaker. 2 folds x 3 seeds.
PROTOCOLS = {
    "sd": {"runs": [(seed, 0) for seed in range(5)]},
    "si": {"runs": [(seed, fold) for fold in (0, 1) for seed in range(3)]},
    # rand: clip-level random stratified 70/15/15 split (literature-comparable, optimistic). 5 seeds.
    "rand": {"runs": [(seed, 0) for seed in range(5)]},
}
