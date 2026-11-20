"""Pre-registration 2 (analysis/PREREGISTRATION_si_capacity.md): the non-trained parts of the study.

1. Logistic-regression rows L1, L1n, L2n per SI direction -> Results/capacity_lr_predictions.csv
2. Inductive normalisation (I): the test speaker's statistics come only from 40 calibration words
   (numpy seed 7); every model is scored on the other 160 words.
   * L1n with calibration statistics vs. transductive statistics.
   * B2 (BiGRU + speaker norm) checkpoints in RUNS_DIR, if present: frame statistics from the
     calibration clips vs. the transductive ones -> Results/capacity_inductive_neural.csv

    python analysis/capacity_inductive.py            # needs the WavLM feature cache (+ B2 checkpoints)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.layer_probe import pooled, spk_norm  # noqa: E402
from analysis.si_linear import clf  # noqa: E402
from common.config import EMOTIONS, RESULTS_DIR, RUNS_DIR, SPEAKERS  # noqa: E402
from common.data import load_index, make_split  # noqa: E402

C_FIXED = 0.001     # pre-registration 1 selected this value in both directions; fixed for L2n
CALIB_SEED, N_CALIB = 7, 40


def calibration_words(df) -> set[str]:
    words = np.array(sorted(df.word.unique()))
    return set(np.random.default_rng(CALIB_SEED).choice(words, N_CALIB, replace=False))


def norm_with(X, spk, stats):
    X = X.copy()
    for s, (m, sd) in stats.items():
        X[spk == s] = (X[spk == s] - m) / (sd + 1e-6)
    return X


def lr_part(df, calib):
    y, spk = df.label.values, df.speaker.values
    P = pooled("wavlm")
    D = P.shape[2] // 2
    X_ms = P.reshape(len(P), -1)
    X_m = P[:, :, :D].reshape(len(P), -1)
    rows = []
    for fold in (0, 1):
        s = make_split(df, "si", 0, fold)
        tr, te = np.concatenate([s["train"], s["val"]]), s["test"]
        tr_spk, te_spk = SPEAKERS[fold], SPEAKERS[1 - fold]
        is_cal = df.word.isin(calib).values
        variants = {"L1": X_ms, "L1n": spk_norm(X_ms, spk), "L2n": spk_norm(X_m, spk)}
        # inductive: training speaker uses all its clips, test speaker only its calibration clips
        stats = {tr_spk: (X_ms[spk == tr_spk].mean(0), X_ms[spk == tr_spk].std(0)),
                 te_spk: (X_ms[(spk == te_spk) & is_cal].mean(0), X_ms[(spk == te_spk) & is_cal].std(0))}
        variants["I(L1n)"] = norm_with(X_ms, spk, stats)
        for name, X in variants.items():
            m = clf(C_FIXED).fit(X[tr], y[tr])
            p = m.predict(X[te])
            rows += [dict(model=name, fold=fold, uid=u, true=EMOTIONS[t], pred=EMOTIONS[q], calibration_word=c)
                     for u, t, q, c in zip(df.uid.values[te], y[te], p, is_cal[te])]
            ok = p == y[te]
            print(f"{name:7s} fold{fold}: all {100 * ok.mean():.2f}  | 160 eval words "
                  f"{100 * ok[~is_cal[te]].mean():.2f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "capacity_lr_predictions.csv", index=False)


def neural_part(df, calib):
    import torch
    from torch.utils.data import DataLoader

    from common.engine import FeatureDataset, collate, load_model, predict
    from common.features import VarLenStore, speaker_stats

    var_dir = RUNS_DIR / "speech" / "wavlm_spknorm"
    cks = sorted(var_dir.glob("si_fold*_seed*/best.pt"))
    if not cks:
        print("no B2 checkpoints: skipping the neural inductive check")
        return
    store = VarLenStore("wavlm")
    spk = df.speaker.values
    is_cal = df.word.isin(calib).values
    trans = speaker_stats("wavlm", spk)
    rows = []
    for ck in cks:
        fold, seed = int(ck.parent.name.split("_")[1][4:]), int(ck.parent.name.split("_")[2][4:])
        te_spk = SPEAKERS[1 - fold]
        acc = acc2 = None
        n = 0
        for i in np.flatnonzero((spk == te_spk) & is_cal):
            x = store[int(i)].astype(np.float64)
            acc = x.sum(0) if acc is None else acc + x.sum(0)
            acc2 = (x ** 2).sum(0) if acc2 is None else acc2 + (x ** 2).sum(0)
            n += len(x)
        mean = acc / n
        ind = dict(trans)
        ind[te_spk] = (mean.astype(np.float32), np.sqrt(np.maximum(acc2 / n - mean ** 2, 1e-8)).astype(np.float32))
        cfg = dict(pipeline="speech", variant="wavlm_spknorm", protocol="si", seed=seed, fold=fold)
        model, device = load_model(cfg)
        test = make_split(df, "si", seed, fold)["test"]
        test = test[~is_cal[test]]
        for kind, stats in (("transductive", trans), ("inductive", ind)):
            ds = FeatureDataset(test, df.label.values, {"speech": store}, False, {"speech": (spk, stats)})
            dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=lambda b: collate(b, ["speech"]))
            probs, y, idx, _, _ = predict(model, dl, device)
            rows.append(dict(fold=fold, seed=seed, stats=kind, accuracy=float((probs.argmax(1) == y).mean()),
                             n=len(y)))
            print(rows[-1], flush=True)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "capacity_inductive_neural.csv", index=False)


def main():
    df = load_index()
    calib = calibration_words(df)
    pd.Series(sorted(calib), name="calibration_word").to_csv(RESULTS_DIR / "capacity_calibration_words.csv",
                                                            index=False)
    lr_part(df, calib)
    neural_part(df, calib)


if __name__ == "__main__":
    main()
