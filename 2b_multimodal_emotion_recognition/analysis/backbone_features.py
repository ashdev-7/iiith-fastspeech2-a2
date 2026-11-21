"""Pre-registration 4 (analysis/PREREGISTRATION_si_backbones.md): pooled features of the frozen encoders.

For every clip and for its 0.9x / 1.1x speed-perturbed copies (common/audio.py:speed_perturb), each encoder's
hidden states are averaged over frame pairs (40 ms, as for the WavLM-Base+ store) and pooled to mean + std
per kept layer -> CACHE_DIR/pooled_<encoder>_<copy>.npy, shape (clips, layers, 2 x dim), float32.

    wavlm_large    microsoft/wavlm-large            hidden states 0, 2, ..., 24
    hubert_large   facebook/hubert-large-ll60k      hidden states 0, 2, ..., 24
    e2v_base       emotion2vec/emotion2vec_base     final layer (funasr)
    e2v_plus_seed  emotion2vec/emotion2vec_plus_seed  LEAKAGE CONTROL ONLY: final layer, plus its own
                   classifier scores on the original clips -> CACHE_DIR/e2v_plus_seed_zeroshot.npy (+ labels)

The encoders never see a label. One clip at a time, no padding (as in common/features.py).

    python analysis/backbone_features.py [--encoders ...] [--limit N --out_dir DIR]   # limit/out_dir: smoke tests
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.audio import SPEED_FACTORS, speed_perturb  # noqa: E402
from common.config import CACHE_DIR, SAMPLE_RATE  # noqa: E402
from common.data import load_index  # noqa: E402

LARGE_LAYERS = tuple(range(0, 25, 2))
ENCODERS = {
    "wavlm_large": ("hf", "microsoft/wavlm-large"),
    "hubert_large": ("hf", "facebook/hubert-large-ll60k"),
    "e2v_base": ("funasr", "emotion2vec/emotion2vec_base"),
    "e2v_plus_seed": ("funasr", "emotion2vec/emotion2vec_plus_seed"),
}
COPIES = {"orig": None, **{f"sp{int(round(100 * f)):03d}": f for f in SPEED_FACTORS}}


def pool_stats(x: np.ndarray) -> np.ndarray:
    """(T, L, D) -> (L, 2D): average frame pairs (40 ms, like the WavLM-Base+ store), then mean and std."""
    x = x.astype(np.float32)
    if x.shape[0] >= 2:
        T = x.shape[0] // 2
        x = x[:2 * T].reshape(T, 2, *x.shape[1:]).mean(1)
    return np.concatenate([x.mean(0), x.std(0)], -1)


class HFEncoder:
    def __init__(self, repo, device):
        from transformers import AutoFeatureExtractor, AutoModel
        self.fe = AutoFeatureExtractor.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, w):
        inp = self.fe(w, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(self.device)
        hs = self.model(inp, output_hidden_states=True).hidden_states
        return torch.stack([hs[i] for i in LARGE_LAYERS], 2)[0].float().cpu().numpy()     # (T, 13, 1024)


class E2VEncoder:
    def __init__(self, repo, device):
        from funasr import AutoModel
        self.m = AutoModel(model=repo, hub="hf", disable_update=True, device=device)

    def __call__(self, w):
        r = self.m.generate(w.astype(np.float32), granularity="frame", extract_embedding=True, disable_pbar=True)
        return np.asarray(r[0]["feats"], dtype=np.float32)[:, None, :]                   # (T, 1, 768)

    def scores(self, w):
        r = self.m.generate(w.astype(np.float32), granularity="utterance", extract_embedding=False,
                            disable_pbar=True)[0]
        return np.asarray(r["scores"], dtype=np.float32), list(r["labels"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_dir", default=str(CACHE_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from prepare_data import load_waves
    df = load_index()
    waves = load_waves(df)
    if args.limit:
        waves = waves[:args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for enc in args.encoders:
        kind, repo = ENCODERS[enc]
        model = (HFEncoder if kind == "hf" else E2VEncoder)(repo, device)
        for copy, f in COPIES.items():
            t = time.time()
            P = np.stack([pool_stats(model(w if f is None else speed_perturb(w, f))) for w in waves])
            np.save(out / f"pooled_{enc}_{copy}.npy", P)
            print(f"{enc} {copy}: {P.shape} in {time.time() - t:.0f} s", flush=True)
        if enc == "e2v_plus_seed":
            res = [model.scores(w) for w in waves]
            np.save(out / "e2v_plus_seed_zeroshot.npy", np.stack([s for s, _ in res]))
            (out / "e2v_plus_seed_zeroshot_labels.txt").write_text("\n".join(res[0][1]), encoding="utf-8")
            print("e2v_plus_seed zero-shot scores saved", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
