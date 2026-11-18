"""Hand-crafted prosodic / spectral baseline (no deep learning).

Utterance-level functionals in the spirit of eGeMAPS: duration, F0 (median, mean,
std, 5/95 pct, range, slope), voiced fraction, energy (mean, std, max, range),
spectral centroid / rolloff / flatness, ZCR, 13 MFCC means+stds -> 44 features.
Classifiers: logistic regression and RBF-SVM, plus a duration-only model to expose
the duration shortcut in TESS. Same SD / SI / rand splits as the neural models.

    python analysis/prosody_baseline.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from common.config import CACHE_DIR, PROTOCOLS, RESULTS_DIR, SAMPLE_RATE  # noqa: E402
from common.data import load_index, make_split  # noqa: E402

SR, HOP = SAMPLE_RATE, 160


def functionals(y: np.ndarray) -> np.ndarray:
    import librosa  # only needed to compute features, not to report them
    f0 = librosa.yin(y, fmin=75, fmax=600, sr=SR, frame_length=1024, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]
    voiced = rms > 0.1 * rms.max()
    v = f0[voiced] if voiced.sum() > 2 else f0
    e = 20 * np.log10(rms + 1e-6)
    t = np.arange(len(v))
    slope = np.polyfit(t, v, 1)[0] if len(v) > 2 else 0.0
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=HOP))
    cent = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    roll = librosa.feature.spectral_rolloff(S=S, sr=SR)[0]
    flat = librosa.feature.spectral_flatness(S=S)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=HOP)[0]
    mf = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, hop_length=HOP)
    feats = [len(y) / SR, np.median(v), v.mean(), v.std(), np.percentile(v, 5), np.percentile(v, 95),
             np.percentile(v, 95) - np.percentile(v, 5), slope, voiced.mean(),
             e.mean(), e.std(), e.max(), e.max() - np.percentile(e, 10),
             cent.mean(), cent.std(), roll.mean(), flat.mean(), zcr.mean()]
    return np.concatenate([np.array(feats, dtype=np.float64), mf.mean(1), mf.std(1)])


def spk_norm(X, spk):
    X = X.copy()
    for s in np.unique(spk):
        m = spk == s
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-9)
    return X


def main():
    df = load_index()
    cache = CACHE_DIR / "prosody_functionals.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        waves = pickle.loads((CACHE_DIR / "waves_16k_trimmed.pkl").read_bytes())
        X = np.stack([functionals(w) for w in waves])
        np.save(cache, X)
    X = np.nan_to_num(X)
    y, spk = df.label.values, df.speaker.values
    models = {
        "duration only (LogReg)": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), [0]),
        "44 functionals (LogReg)": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)), None),
        "44 functionals (RBF-SVM)": (lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")), None),
    }
    rows = []
    for name, (mk, cols) in models.items():
        for norm in (False, True):
            Xu = spk_norm(X, spk) if norm else X
            Xu = Xu[:, cols] if cols else Xu
            for proto, spec in PROTOCOLS.items():
                accs = []
                for seed, fold in spec["runs"]:
                    s = make_split(df, proto, seed, fold)
                    tr = np.concatenate([s["train"], s["val"]])
                    clf = mk().fit(Xu[tr], y[tr])
                    accs.append((clf.predict(Xu[s["test"]]) == y[s["test"]]).mean())
                rows.append(dict(model=name, speaker_norm=norm, protocol=proto,
                                 acc_mean=100 * np.mean(accs), acc_std=100 * np.std(accs, ddof=1), runs=len(accs)))
                print(rows[-1], flush=True)
    report(pd.DataFrame(rows))


def report(t: pd.DataFrame):
    t.round(2).to_csv(RESULTS_DIR / "prosody_baseline.csv", index=False)
    wide = t.assign(acc=t.acc_mean.round(2).astype(str) + " ± " + t.acc_std.round(2).astype(str)) \
            .pivot_table(index=["model", "speaker_norm"], columns="protocol", values="acc", aggfunc="first")
    (RESULTS_DIR / "prosody_baseline.md").write_text(
        "# Hand-crafted prosody baseline (test accuracy %, mean ± std over the same splits)\n\n"
        + wide.to_markdown() + "\n", encoding="utf-8")
    print(wide.to_string())


if __name__ == "__main__":
    main()
