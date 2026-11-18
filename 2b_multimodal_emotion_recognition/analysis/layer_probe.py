"""Layer-wise linear probes on the frozen encoders.

For every cached layer we pool each clip to [mean ; std] over time and fit a
logistic-regression probe for (a) emotion under SD and SI and (b) speaker identity.
This shows *where* emotion lives in WavLM, how strongly the same features encode
the speaker (the reason SI is hard), and how much label-free speaker normalisation
removes that confound.

    python analysis/layer_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common.config import PLOTS_DIR, RESULTS_DIR  # noqa: E402
from common.data import load_index, make_split  # noqa: E402
from common.features import VarLenStore  # noqa: E402

STORES = {"wavlm": [f"WavLM L{i}" for i in (0, 2, 4, 6, 8, 10, 12)],
          "logmel": ["log-Mel"],
          "text_gt": [f"DistilRoBERTa L{i}" for i in range(7)]}


def pooled(name: str) -> np.ndarray:
    s = VarLenStore(name)
    out = np.zeros((len(s), s.num_layers, 2 * s.dim), dtype=np.float32)
    for i in range(len(s)):
        x = s[i].astype(np.float32)
        out[i] = np.concatenate([x.mean(0), x.std(0)], -1)
    return out


def spk_norm(X: np.ndarray, spk: np.ndarray) -> np.ndarray:
    X = X.copy()
    for s in np.unique(spk):
        m = spk == s
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-6)
    return X


def probe(X, y, tr, te) -> float:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    clf.fit(X[tr], y[tr])
    return float((clf.predict(X[te]) == y[te]).mean())


def main():
    df = load_index()
    y, spk = df.label.values, df.speaker.values
    spk_y = (spk == "YAF").astype(int)
    sd = [make_split(df, "sd", s) for s in range(5)]
    si = [make_split(df, "si", 0, f) for f in (0, 1)]
    rows = []
    for name, labels in STORES.items():
        try:
            P = pooled(name)
        except FileNotFoundError:
            continue
        for li, lab in enumerate(labels):
            X = P[:, li]
            Xn = spk_norm(X, spk)
            trva = lambda s: np.concatenate([s["train"], s["val"]])
            r = dict(store=name, layer=lab,
                     emotion_sd=np.mean([probe(X, y, trva(s), s["test"]) for s in sd]),
                     emotion_si=np.mean([probe(X, y, trva(s), s["test"]) for s in si]),
                     emotion_si_spknorm=np.mean([probe(Xn, y, trva(s), s["test"]) for s in si]),
                     speaker_id=np.mean([probe(X, spk_y, trva(s), s["test"]) for s in sd]))
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    report(pd.DataFrame(rows))


def report(t: pd.DataFrame):
    t.round(4).to_csv(RESULTS_DIR / "layer_probe.csv", index=False)
    (RESULTS_DIR / "layer_probe.md").write_text(
        "# Linear probes on frozen-encoder layers (pooled mean+std, logistic regression)\n\n"
        "emotion_sd / emotion_si: 7-class accuracy; emotion_si_spknorm: features z-normalised per speaker "
        "(label-free); speaker_id: OAF-vs-YAF accuracy (SD splits).\n\n"
        + (t.assign(**{c: (100 * t[c]).round(1) for c in t.columns if c not in ("store", "layer")})
           .to_markdown(index=False)) + "\n", encoding="utf-8")

    w = t[t.store == "wavlm"]
    if len(w):
        fig, ax = plt.subplots(figsize=(7, 3.4))
        x = np.arange(len(w))
        for col, lab, c in (("emotion_sd", "emotion, SD", "#0072B2"), ("emotion_si", "emotion, SI", "#D55E00"),
                            ("emotion_si_spknorm", "emotion, SI + speaker norm", "#E69F00"),
                            ("speaker_id", "speaker identity", "#7F7F7F")):
            ax.plot(x, 100 * w[col], "o-", label=lab, color=c)
        ax.axhline(100 / 7, ls=":", color="k", lw=1)
        ax.text(0, 100 / 7 + 1.5, "emotion chance", fontsize=7)
        ax.set_xticks(x, w.layer.str.replace("WavLM ", ""))
        ax.set_xlabel("WavLM-Base+ layer")
        ax.set_ylabel("probe accuracy (%)")
        ax.set_ylim(0, 102)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title("Linear probes on frozen WavLM layers (pooled mean+std)", fontsize=9)
        fig.savefig(PLOTS_DIR / "layer_probe_wavlm.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
