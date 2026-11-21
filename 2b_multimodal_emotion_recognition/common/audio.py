"""Speech preprocessing: resample, mono, trim silence, normalise."""
from __future__ import annotations

import librosa
import numpy as np

from .config import MAX_AUDIO_S, SAMPLE_RATE, TRIM_PAD_S, TRIM_TOP_DB


def load_and_preprocess(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a clip and return a float32 waveform in [-1, 1].

    1. Resample 24.4 kHz -> 16 kHz (the rate WavLM was pre-trained on; speech
       energy relevant to emotion is far below the 8 kHz Nyquist limit).
    2. Remove DC offset.
    3. Trim leading/trailing silence (30 dB below peak) keeping 50 ms margin, so
       frames fed to the temporal model contain speech rather than room noise.
    4. Peak-normalise to 0.95. Loudness differences between recordings are
       removed; *within*-utterance energy dynamics (an emotion cue) are kept.
    5. Cap at 4 s (no TESS clip reaches it; guards against corrupt inputs).
    Variable length is preserved: batching pads and masks downstream.
    """
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = y - np.mean(y)
    _, (start, end) = librosa.effects.trim(y, top_db=TRIM_TOP_DB, frame_length=512, hop_length=128)
    pad = int(TRIM_PAD_S * sr)
    y = y[max(0, start - pad): min(len(y), end + pad)]
    y = y[: int(MAX_AUDIO_S * sr)]
    peak = np.max(np.abs(y)) + 1e-9
    return (0.95 * y / peak).astype(np.float32)


SPEED_FACTORS = (0.9, 1.1)


def speed_perturb(y: np.ndarray, factor: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Kaldi/sox-style speed perturbation: resample so that tempo AND pitch change by `factor`
    (factor 1.1 -> 10 % faster and higher; the clip becomes 1/factor as long), then peak-normalise
    to 0.95 like the original preprocessing. Used only for training-speaker augmentation
    (analysis/PREREGISTRATION_si_augmentation.md)."""
    z = librosa.resample(y.astype(np.float32), orig_sr=int(round(sr * factor)), target_sr=sr)
    return (0.95 * z / (np.max(np.abs(z)) + 1e-9)).astype(np.float32)


def speed_store_name(factor: float) -> str:
    return f"wavlm_sp{int(round(100 * factor)):03d}"


def log_mel(y: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = 80) -> np.ndarray:
    """80-d log-Mel filterbank, 25 ms window / 10 ms hop -> (frames, 80)."""
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, win_length=400, hop_length=160,
                                       n_mels=n_mels, fmin=20, fmax=sr // 2)
    return np.log(m + 1e-6).T.astype(np.float32)
