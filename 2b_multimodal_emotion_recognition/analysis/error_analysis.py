"""Error analysis: pick 3-5 systematic failure cases and explain them acoustically.

A case is a clip misclassified by the multimodal model in *every* seed of the
speaker-independent protocol (so the error is a property of the clip, not of a
lucky initialisation). For each case we compare its prosody (F0 level/range,
energy, duration) with the same speaker's distribution for the true and the
predicted emotion.

    python analysis/error_analysis.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.audio import log_mel  # noqa: E402
from common.config import CACHE_DIR, EMOTION_NAMES, EMOTIONS, PLOTS_DIR, RESULTS_DIR, RUNS_DIR, SAMPLE_RATE  # noqa: E402
from common.data import load_index  # noqa: E402

MODEL = ("fusion", "gated_wavlm_gt")
N_CASES = 5


def prosody(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    f0 = librosa.yin(y, fmin=75, fmax=600, sr=sr, frame_length=1024, hop_length=160)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=160)[0]
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]
    voiced = rms > 0.1 * rms.max()
    v = f0[voiced]
    return dict(dur_s=len(y) / sr, f0_mean=float(np.median(v)) if len(v) else np.nan,
                f0_range=float(np.percentile(v, 95) - np.percentile(v, 5)) if len(v) else np.nan,
                rms_db=float(20 * np.log10(rms[voiced].mean() + 1e-9)), f0_track=np.where(voiced, f0, np.nan))


def main():
    df = load_index()
    waves = pickle.loads((CACHE_DIR / "waves_16k_trimmed.pkl").read_bytes())
    uid2i = {u: i for i, u in enumerate(df.uid)}
    preds = []
    for f in sorted((RUNS_DIR / MODEL[0] / MODEL[1]).glob("si_*/test_predictions.csv")):
        p = pd.read_csv(f)
        p["run"] = f.parent.name
        preds.append(p)
    P = pd.concat(preds)
    pcols = [f"p_{e}" for e in EMOTIONS]
    agg = P.groupby("uid").agg(true=("true", "first"), speaker=("speaker", "first"), word=("word", "first"),
                               n_runs=("run", "count"), **{c: (c, "mean") for c in pcols})
    agg["n_wrong"] = P.assign(w=P.pred != P.true).groupby("uid").w.sum()
    agg["pred_mode"] = P.groupby("uid").pred.agg(lambda s: s.value_counts().index[0])
    agg["conf"] = agg[pcols].max(1)
    sys_err = agg[(agg.n_wrong == agg.n_runs) & (agg.pred_mode != agg.true)].sort_values("conf", ascending=False)
    print(f"{len(sys_err)} clips misclassified in every SI seed (of {len(agg)})")
    pair_counts = sys_err.groupby(["true", "pred_mode"]).size().sort_values(ascending=False)
    print(pair_counts.head(10))

    # one case per distinct confusion pair, most frequent pairs first, highest confidence within the pair
    cases = []
    for (t, p), _ in pair_counts.items():
        cases.append(sys_err[(sys_err.true == t) & (sys_err.pred_mode == p)].index[0])
        if len(cases) == N_CASES:
            break

    # speaker x emotion prosody reference distributions (only the classes we need)
    need = {(sys_err.loc[c, "speaker"], e) for c in cases for e in (sys_err.loc[c, "true"], sys_err.loc[c, "pred_mode"])}
    ref = {}
    for spk, emo in sorted(need):
        rows = df[(df.speaker == spk) & (df.emotion == emo)].index
        st = pd.DataFrame([{k: v for k, v in prosody(waves[i]).items() if k != "f0_track"} for i in rows])
        ref[(spk, emo)] = st.mean()

    table, fig_rows = [], []
    for c in cases:
        r = sys_err.loc[c]
        i = uid2i[c]
        pr = prosody(waves[i])
        rt, rp = ref[(r.speaker, r.true)], ref[(r.speaker, r.pred_mode)]
        top = r[pcols].astype(float).sort_values(ascending=False).head(3)
        table.append({
            "clip": c, "speaker": r.speaker, "true": EMOTION_NAMES[r.true], "predicted": EMOTION_NAMES[r.pred_mode],
            "wrong in": f"{int(r.n_wrong)}/{int(r.n_runs)} seeds", "mean p(pred)": f"{r[f'p_{r.pred_mode}']:.2f}",
            "mean p(true)": f"{r[f'p_{r.true}']:.2f}",
            "F0 median Hz (clip | true-class | pred-class)": f"{pr['f0_mean']:.0f} | {rt.f0_mean:.0f} | {rp.f0_mean:.0f}",
            "F0 range Hz": f"{pr['f0_range']:.0f} | {rt.f0_range:.0f} | {rp.f0_range:.0f}",
            "energy dB": f"{pr['rms_db']:.1f} | {rt.rms_db:.1f} | {rp.rms_db:.1f}",
            "duration s": f"{pr['dur_s']:.2f} | {rt.dur_s:.2f} | {rp.dur_s:.2f}",
            "top-3": ", ".join(f"{EMOTION_NAMES[k[2:]]} {v:.2f}" for k, v in top.items()),
        })
        fig_rows.append((c, r, waves[i], pr))

    t = pd.DataFrame(table)
    t.to_csv(RESULTS_DIR / "error_analysis_cases.csv", index=False)
    md = ["# Error analysis: systematic failures of the gated-fusion model (speaker-independent)", "",
          f"{len(sys_err)} of {len(agg)} test clips are misclassified in every seed. Most frequent confusion pairs:", "",
          pair_counts.rename("clips").reset_index().replace(EMOTION_NAMES).head(10).to_markdown(index=False), "",
          "Selected cases (prosody columns: clip | same speaker's mean for the true class | for the predicted class):", "",
          t.to_markdown(index=False), ""]
    (RESULTS_DIR / "error_analysis.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    fig, axes = plt.subplots(len(fig_rows), 2, figsize=(12, 2.4 * len(fig_rows)),
                             gridspec_kw=dict(width_ratios=[3, 1.2]))
    for ax_row, (c, r, y, pr) in zip(np.atleast_2d(axes), fig_rows):
        M = log_mel(y).T
        ax = ax_row[0]
        ax.imshow(M, origin="lower", aspect="auto", cmap="magma", extent=(0, len(y) / SAMPLE_RATE, 0, 80))
        ax2 = ax.twinx()
        tt = np.arange(len(pr["f0_track"])) * 160 / SAMPLE_RATE
        ax2.plot(tt, pr["f0_track"], color="#56B4E9", lw=1.5)
        ax2.set_ylim(75, 600)
        ax2.set_ylabel("F0 (Hz)", fontsize=7)
        ax.set_ylabel("mel bin", fontsize=7)
        ax.set_title(f"{c}: true {EMOTION_NAMES[r.true]}  →  predicted {EMOTION_NAMES[r.pred_mode]}", fontsize=9)
        probs = r[[f"p_{e}" for e in EMOTIONS]].astype(float).values
        colors = ["#D55E00" if e == r.pred_mode else "#009E73" if e == r.true else "#BBBBBB" for e in EMOTIONS]
        ax_row[1].barh(range(7), probs, color=colors)
        ax_row[1].set_yticks(range(7), [EMOTION_NAMES[e] for e in EMOTIONS], fontsize=7)
        ax_row[1].set_xlim(0, 1)
        ax_row[1].invert_yaxis()
        ax_row[1].set_xlabel("mean prob. over seeds", fontsize=7)
    np.atleast_2d(axes)[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "error_analysis_cases.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
