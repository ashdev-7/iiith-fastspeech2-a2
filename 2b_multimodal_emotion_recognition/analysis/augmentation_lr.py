"""Pre-registration 3 (analysis/PREREGISTRATION_si_augmentation.md): the logistic-regression arm.

L1n   = pre-registered model of study 2: pooled mean+std of all 7 WavLM layers, per-speaker z-scoring of the
        pooled vectors (label-free †), StandardScaler + L2 logistic regression, C = 0.001.
A-L1n = the same, trained on the training speaker's original clips plus their 0.9x and 1.1x speed-perturbed
        copies. Each perturbed copy is z-scored with its own statistics (a pseudo-speaker). Test = original
        audio of the unseen speaker, normalised exactly as for L1n.

    python analysis/augmentation_lr.py   # needs the wavlm, wavlm_sp090, wavlm_sp110 stores
    -> Results/augmentation_lr_predictions.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.layer_probe import pooled, spk_norm  # noqa: E402
from analysis.si_linear import clf  # noqa: E402
from common.audio import SPEED_FACTORS, speed_store_name  # noqa: E402
from common.config import EMOTIONS, RESULTS_DIR, SPEAKERS  # noqa: E402
from common.data import load_index, make_split  # noqa: E402

C_FIXED = 0.001


def flat(name: str) -> np.ndarray:
    P = pooled(name)
    return P.reshape(len(P), -1)


def main():
    df = load_index()
    y, spk = df.label.values, df.speaker.values
    X = spk_norm(flat("wavlm"), spk)
    Xa = [spk_norm(flat(speed_store_name(f)), spk) for f in SPEED_FACTORS]   # own stats per copy and speaker
    rows = []
    for fold in (0, 1):
        s = make_split(df, "si", 0, fold)
        tr, te = np.concatenate([s["train"], s["val"]]), s["test"]
        assert set(spk[tr]) == {SPEAKERS[fold]} and set(spk[te]) == {SPEAKERS[1 - fold]}
        fits = {"L1n": (X[tr], y[tr]),
                "A-L1n": (np.concatenate([X[tr]] + [A[tr] for A in Xa]), np.concatenate([y[tr]] * (1 + len(Xa))))}
        for name, (Xtr, ytr) in fits.items():
            p = clf(C_FIXED).fit(Xtr, ytr).predict(X[te])
            rows += [dict(model=name, fold=fold, uid=u, true=EMOTIONS[t], pred=EMOTIONS[q])
                     for u, t, q in zip(df.uid.values[te], y[te], p)]
            print(f"{name:6s} {SPEAKERS[fold]}->{SPEAKERS[1 - fold]}: {100 * (p == y[te]).mean():.2f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "augmentation_lr_predictions.csv", index=False)


if __name__ == "__main__":
    main()
