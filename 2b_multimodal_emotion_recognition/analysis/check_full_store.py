"""Consistency check between the two WavLM feature stores.

The main store ("wavlm") keeps layers 0,2,...,12 and averages pairs of 20 ms frames; the
ablation store ("wavlm_full") keeps all 13 layers at 20 ms. Pooling the full store's even
layers must therefore reproduce the main store, up to fp16 rounding (the main store pools in
fp32 before casting, the full store is cast first). Confirms both stores come from the same
audio, alignment and model, so the ablation changes only resolution and depth.

    python analysis/check_full_store.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from common.config import RESULTS_DIR  # noqa: E402
from common.features import SSL_KEEP_LAYERS, SSL_TIME_POOL, VarLenStore  # noqa: E402


def main(n: int = 200):
    full, main_ = VarLenStore("wavlm_full"), VarLenStore("wavlm")
    rng = np.random.default_rng(0)
    rel, lens = [], 0
    for i in rng.choice(len(main_), size=min(n, len(main_)), replace=False):
        a = full[int(i)].astype(np.float32)[:, list(SSL_KEEP_LAYERS)]
        b = main_[int(i)].astype(np.float32)
        T = b.shape[0]
        p = a[: T * SSL_TIME_POOL].reshape(T, SSL_TIME_POOL, *a.shape[1:]).mean(1)
        rel.append(np.abs(p - b).max() / (np.abs(b).max() + 1e-9))
        lens += int(full[int(i)].shape[0] // SSL_TIME_POOL == T)
    msg = (f"wavlm_full: {full.num_layers} layers x {full.dim}; wavlm: {main_.num_layers} layers x {main_.dim}\n"
           f"{len(rel)} random clips: frame counts consistent for {lens}/{len(rel)}; "
           f"max relative deviation of pooled(full) vs main = {max(rel):.2e} (fp16 rounding ~1e-3)\n")
    print(msg)
    (RESULTS_DIR / "ablation_store_check.txt").write_text(msg, encoding="utf-8")
    assert lens == len(rel) and max(rel) < 1e-2, "stores are inconsistent"


if __name__ == "__main__":
    main()
