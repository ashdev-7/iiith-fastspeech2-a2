"""Feature extraction + an on-disk variable-length feature store.

Speech  : frozen WavLM-Base+ -> all 13 hidden layers, (time_steps x 13 x 768), fp16.
          (A learnable softmax layer-weighting is trained downstream, SUPERB-style.)
          Baseline: 80-d log-Mel (time_steps x 1 x 80).
Text    : frozen emotion-tuned DistilRoBERTa -> all 7 hidden layers, (tokens x 7 x 768).

Features are stored as one concatenated fp16 memmap + an offsets array, so the
~3.5 GB of WavLM states never have to fit into RAM at once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import CACHE_DIR, SAMPLE_RATE, SPEECH_SSL_MODEL, TEXT_PLM_MODEL


class VarLenStore:
    """Read-only view over `<name>.data.npy` (sum_T, L, D) and `<name>.offsets.npy` (N+1,)."""

    def __init__(self, name: str, cache_dir: Path = CACHE_DIR):
        self.name = name
        self.path = cache_dir / f"{name}.data.npy"
        mm = np.load(self.path, mmap_mode="r")
        self.header_bytes, self.num_layers, self.dim = mm.offset, *mm.shape[1:]
        self.row_bytes = self.num_layers * self.dim * 2  # fp16
        del mm
        self.offsets = np.load(cache_dir / f"{name}.offsets.npy")
        self._fh = None

    def __len__(self):
        return len(self.offsets) - 1

    def length(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    def __getitem__(self, i: int) -> np.ndarray:
        # one contiguous read per utterance (much faster than memmap page faults on Windows)
        if self._fh is None:
            self._fh = open(self.path, "rb", buffering=0)
        T = self.length(i)
        buf = np.empty((T, self.num_layers, self.dim), dtype=np.float16)
        self._fh.seek(self.header_bytes + int(self.offsets[i]) * self.row_bytes)
        self._fh.readinto(memoryview(buf).cast("B"))
        return buf


def write_store(name: str, arrays_iter, n_items: int, total_len: int, num_layers: int, dim: int,
                cache_dir: Path = CACHE_DIR):
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = np.lib.format.open_memmap(cache_dir / f"{name}.data.npy", mode="w+", dtype=np.float16,
                                     shape=(total_len, num_layers, dim))
    offsets = np.zeros(n_items + 1, dtype=np.int64)
    pos = 0
    for i, arr in enumerate(arrays_iter):
        assert arr.shape[1:] == (num_layers, dim), arr.shape
        data[pos: pos + len(arr)] = arr.astype(np.float16)
        pos += len(arr)
        offsets[i + 1] = pos
    assert pos == total_len, (pos, total_len)
    data.flush()
    np.save(cache_dir / f"{name}.offsets.npy", offsets)


def speaker_stats(name: str, speakers: np.ndarray, cache_dir: Path = CACHE_DIR) -> dict:
    """Per-speaker mean / std of every (layer, dim) over all frames of that speaker.

    Label-free speaker normalisation (speaker-level CMVN). Under the SI protocol this
    uses the *unlabelled* audio of the test speaker, i.e. it is transductive
    adaptation; it is reported as a separate, clearly marked variant.
    """
    path = cache_dir / f"{name}.spkstats.npz"
    if path.exists():
        z = np.load(path)
        return {s: (z[f"{s}_mean"], z[f"{s}_std"]) for s in np.unique(speakers)}
    store = VarLenStore(name, cache_dir)
    out = {}
    for s in np.unique(speakers):
        acc = acc2 = None
        n = 0
        for i in np.flatnonzero(speakers == s):
            x = store[int(i)].astype(np.float64)
            acc = x.sum(0) if acc is None else acc + x.sum(0)
            acc2 = (x ** 2).sum(0) if acc2 is None else acc2 + (x ** 2).sum(0)
            n += len(x)
        mean = acc / n
        std = np.sqrt(np.maximum(acc2 / n - mean ** 2, 1e-8))
        out[s] = (mean.astype(np.float32), std.astype(np.float32))
    np.savez(path, **{f"{s}_mean": m for s, (m, _) in out.items()}, **{f"{s}_std": v for s, (_, v) in out.items()})
    return out


# ------------------------------------------------------------------ speech
def ssl_num_frames(n_samples: int) -> int:
    """Output length of the WavLM/wav2vec2 CNN front-end (kernel/stride stack)."""
    for k, s in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
        n_samples = (n_samples - k) // s + 1
    return n_samples


SSL_KEEP_LAYERS = (0, 2, 4, 6, 8, 10, 12)  # CNN output + every 2nd transformer layer
SSL_TIME_POOL = 2                          # 20 ms -> 40 ms frames
SSL_FULL_LAYERS = tuple(range(13))         # ablation "wavlm_full": every layer, native 20 ms frames


def _pooled_len(n_frames: int, pool: int = SSL_TIME_POOL) -> int:
    return max(1, n_frames // pool)


@torch.no_grad()
def extract_speech_ssl(waves: list[np.ndarray], name: str = "wavlm", device: str = "cpu",
                       layers: tuple = SSL_KEEP_LAYERS, pool: int = SSL_TIME_POOL):
    """Frozen WavLM-Base+ hidden states -> (T/2, 7, 768) fp16 per utterance.

    Storage budget: all 13 layers at 50 Hz would be ~5.6 GB for TESS, which
    cannot be held in RAM on an 8 GB machine. Adjacent transformer layers are
    highly redundant (high CKA similarity), so we keep every second layer
    (still spanning the full depth for the learnable layer weighting) and
    average-pool pairs of frames (40 ms resolution is ample for prosodic
    contours; each frame embedding already encodes ~400 ms of context).
    Result: ~1.5 GB. The ablation store "wavlm_full" (layers=SSL_FULL_LAYERS, pool=1)
    keeps everything and tests whether this reduction costs accuracy.
    """
    from transformers import AutoFeatureExtractor, WavLMModel

    fe = AutoFeatureExtractor.from_pretrained(SPEECH_SSL_MODEL)
    model = WavLMModel.from_pretrained(SPEECH_SSL_MODEL).to(device).eval()
    dim = model.config.hidden_size
    total = sum(_pooled_len(ssl_num_frames(len(w)), pool) for w in waves)

    def gen():
        # one utterance at a time: no padding, so outputs are exactly what the
        # model would compute on the clip in isolation (group-norm front-end is padding-sensitive)
        for w in tqdm(waves, desc="WavLM", mininterval=30):
            inp = fe(w, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
            hs = model(inp, output_hidden_states=True).hidden_states       # tuple(13) of (1,T,D)
            x = torch.stack([hs[i] for i in layers], dim=2)[0]              # (T, L, D)
            T = _pooled_len(x.shape[0], pool)
            if pool > 1 and x.shape[0] >= pool:
                x = x[: T * pool].reshape(T, pool, *x.shape[1:]).mean(1)
            yield x.float().cpu().numpy()

    write_store(name, gen(), len(waves), total, len(layers), dim)


def extract_logmel(waves: list[np.ndarray], name: str = "logmel"):
    from .audio import log_mel
    feats = [log_mel(w)[:, None, :] for w in tqdm(waves, desc="log-Mel", mininterval=30)]
    write_store(name, iter(feats), len(feats), sum(len(f) for f in feats), 1, feats[0].shape[-1])


# ------------------------------------------------------------------ text
@torch.no_grad()
def extract_text(transcripts: list[str], name: str, device: str = "cpu") -> np.ndarray:
    """Token-level hidden states of the frozen PLM + its zero-shot class probabilities.

    Returns the zero-shot probability matrix (N, 7) in the PLM's own label order.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEXT_PLM_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(TEXT_PLM_MODEL).to(device).eval()
    uniq = sorted(set(transcripts))
    cache, probs = {}, {}
    for t in tqdm(uniq, desc=f"text[{name}]", mininterval=30):
        enc = tok(t, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        cache[t] = torch.stack(out.hidden_states, dim=2)[0].float().cpu().numpy()  # (tokens, 7, 768)
        probs[t] = out.logits.softmax(-1)[0].cpu().numpy()
    feats = [cache[t] for t in transcripts]
    write_store(name, iter(feats), len(feats), sum(len(f) for f in feats), *feats[0].shape[1:])
    zs = np.stack([probs[t] for t in transcripts])
    np.save(CACHE_DIR / f"{name}.zeroshot_probs.npy", zs)
    labels = [model.config.id2label[i] for i in range(zs.shape[1])]
    (CACHE_DIR / f"{name}.zeroshot_labels.txt").write_text("\n".join(labels))
    return zs
