"""Pre-registration 5 (analysis/PREREGISTRATION_si_final.md): pooled features of the two frozen encoders.

Copies of every clip: the original, the 0.9x / 1.1x speed-perturbed copies (common/audio.py:speed_perturb) and four
pitch-shifted copies at -4, -2, +2, +4 semitones with tempo kept (common/audio.py:pitch_shift). Each encoder's hidden
states are averaged over frame pairs (40 ms) and pooled to mean + std per kept layer (backbone_features.pool_stats)
-> CACHE_DIR/pooled_<encoder>_<copy>.npy, shape (clips, layers, 2 x dim), float32.

    wavlm_large    microsoft/wavlm-large      hidden states 0, 2, ..., 24  (as in pre-registration 4)
    whisper_large  openai/whisper-large-v3    ENCODER only, hidden states 0, 2, ..., 32; float16 on GPU.
                   Whisper's front end pads every clip to 30 s; only the encoder frames that cover the clip
                   itself (ceil(samples / 320), 20 ms each) are kept.

The encoders never see a label. The analysis reads the perturbed copies of training rows only (audit D4).

    python analysis/final_features.py [--encoders ...] [--whisper_repo R] [--limit N --out_dir DIR]   # smoke tests
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.backbone_features import HFEncoder, pool_stats  # noqa: E402
from common.audio import PITCH_STEPS, SPEED_FACTORS, pitch_shift, speed_perturb  # noqa: E402
from common.config import CACHE_DIR, SAMPLE_RATE  # noqa: E402
from common.data import load_index  # noqa: E402


def pitch_name(k: int) -> str:
    return f"pitch{'m' if k < 0 else 'p'}{abs(k)}"


COPIES = {"orig": lambda w: w,
          **{f"sp{int(round(100 * f)):03d}": (lambda w, f=f: speed_perturb(w, f)) for f in SPEED_FACTORS},
          **{pitch_name(k): (lambda w, k=k: pitch_shift(w, k)) for k in PITCH_STEPS}}


class WhisperEncoder:
    """Frozen Whisper encoder; returns (T, layers, dim) for the frames that cover the clip."""

    def __init__(self, repo, device):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.fe = WhisperFeatureExtractor.from_pretrained(repo)
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.enc = WhisperModel.from_pretrained(repo, dtype=self.dtype).get_encoder().to(device).eval()
        self.layers = tuple(range(0, self.enc.config.encoder_layers + 1, 2))
        self.device = device

    @torch.no_grad()
    def __call__(self, w):
        f = self.fe(w, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        hs = self.enc(f.to(self.device, self.dtype), output_hidden_states=True).hidden_states
        n = int(np.ceil(len(w) / (2 * self.fe.hop_length)))        # 2 mel frames (2 x 10 ms) per encoder frame
        out = torch.stack([hs[i][0, :n] for i in self.layers], 1).float().cpu().numpy()
        assert np.isfinite(out).all(), "non-finite Whisper features (float16 overflow?)"
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["wavlm_large", "whisper_large"])
    ap.add_argument("--whisper_repo", default="openai/whisper-large-v3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_dir", default=str(CACHE_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from prepare_data import load_waves
    waves = load_waves(load_index())
    if args.limit:
        waves = waves[:args.limit]
    t = time.time()
    copies = {c: [fn(w) for w in waves] for c, fn in COPIES.items()}     # identical input for both encoders
    print(f"{len(copies)} copies of {len(waves)} clips in {time.time() - t:.0f} s", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for enc in args.encoders:
        model = (HFEncoder("microsoft/wavlm-large", device) if enc == "wavlm_large"
                 else WhisperEncoder(args.whisper_repo, device))
        for copy, ws in copies.items():
            t = time.time()
            P = np.stack([pool_stats(model(w)) for w in ws])
            np.save(out / f"pooled_{enc}_{copy}.npy", P)
            print(f"{enc} {copy}: {P.shape} in {time.time() - t:.0f} s", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
