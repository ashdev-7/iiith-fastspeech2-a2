"""Test-time noise robustness: when does the second modality help?

Additive white Gaussian noise at SNR in {20, 10, 5, 0} dB is mixed into every
(preprocessed) clip, then the *whole* front-end is re-run on the noisy audio:
WavLM features, Whisper transcripts and their DistilRoBERTa features. Trained
checkpoints are evaluated unchanged (no retraining on noise):

  speech-only            : noisy WavLM
  text-only (Whisper)    : transcripts of the noisy audio
  fusion GT transcripts  : noisy WavLM + clean ground-truth text (text is uninformative)
  fusion Whisper         : noisy WavLM + transcripts of the noisy audio

    python analysis/noise_robustness.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from common.config import CACHE_DIR, PLOTS_DIR, RESULTS_DIR, RUNS_DIR  # noqa: E402
from common.data import clean_transcript, load_index  # noqa: E402
from common.engine import evaluate  # noqa: E402
from common.features import extract_speech_ssl, extract_text  # noqa: E402

SNRS = [20, 10, 5, 0]
MODELS = [  # (pipeline, variant, label, needs noisy speech, needs noisy ASR text)
    ("speech", "wavlm", "Speech-only (WavLM)", True, False),
    ("speech", "wavlm_spknorm", "Speech-only + spk-norm", True, False),
    ("text", "asr", "Text-only (Whisper)", False, True),
    ("fusion", "gated_wavlm_gt", "Fusion GMU (GT text)", True, False),
    ("fusion", "gated_wavlm_asr", "Fusion GMU (Whisper text)", True, True),
    ("fusion", "gated_wavlm_asr_spknorm", "Fusion GMU (Whisper) + spk-norm", True, True),
]


def add_noise(y: np.ndarray, snr_db: float, rng) -> np.ndarray:
    p_sig = np.mean(y ** 2)
    noise = rng.standard_normal(len(y)).astype(np.float32) * np.sqrt(p_sig / 10 ** (snr_db / 10))
    z = y + noise
    return (0.95 * z / (np.abs(z).max() + 1e-9)).astype(np.float32)   # same peak normalisation as preprocessing


def build_noisy_features(device: str):
    from prepare_data import transcribe
    waves = pickle.loads((CACHE_DIR / "waves_16k_trimmed.pkl").read_bytes())
    df = load_index()
    for snr in SNRS:
        if (CACHE_DIR / f"text_asr_snr{snr}.offsets.npy").exists():
            continue
        rng = np.random.default_rng(snr)
        noisy = [add_noise(w, snr, rng) for w in waves]
        extract_speech_ssl(noisy, name=f"wavlm_snr{snr}", device=device)
        raw = transcribe(noisy)
        pd.DataFrame({"uid": df.uid, "asr_raw": raw}).to_csv(CACHE_DIR / f"asr_transcripts_snr{snr}.csv", index=False)
        extract_text([clean_transcript(t) for t in raw], name=f"text_asr_snr{snr}", device=device)
        print(f"SNR {snr} dB: features ready; sample ASR: {raw[:3]}", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    build_noisy_features(device)
    rows = []
    for pipe, var, label, ns, nt in MODELS:
        for ck in sorted((RUNS_DIR / pipe / var).glob("*/best.pt")):
            cfg = torch.load(ck, map_location="cpu", weights_only=False)["cfg"]
            if cfg["protocol"] not in ("sd", "si"):
                continue
            model = None
            for snr in [None] + SNRS:
                ov = {}
                if snr is not None and ns:
                    ov["speech"] = f"wavlm_snr{snr}"
                if snr is not None and nt:
                    ov["text"] = f"text_asr_snr{snr}"
                (probs, y, *_), model = evaluate(cfg, ov, model)
                rows.append(dict(model=label, pipeline=pipe, variant=var, protocol=cfg["protocol"],
                                 run=ck.parent.name, snr="clean" if snr is None else snr,
                                 acc=float((probs.argmax(1) == y).mean())))
            print(label, ck.parent.name, [f"{r['acc']:.3f}" for r in rows[-5:]], flush=True)
    report(pd.DataFrame(rows))


def report(t: pd.DataFrame):
    t.to_csv(RESULTS_DIR / "noise_robustness_runs.csv", index=False)
    order = ["clean"] + SNRS
    g = t.groupby(["protocol", "model", "snr"]).acc.agg(["mean", "std"]).reset_index()
    g["cell"] = (100 * g["mean"]).round(1).astype(str) + " ± " + (100 * g["std"]).round(1).astype(str)
    wide = g.pivot_table(index=["protocol", "model"], columns="snr", values="cell", aggfunc="first")[order]
    (RESULTS_DIR / "noise_robustness.md").write_text(
        "# Test-time additive white noise (accuracy %, mean ± std over runs; models trained on clean audio)\n\n"
        + wide.to_markdown() + "\n", encoding="utf-8")
    print(wide.to_string())
    # noisy ASR transcript punctuation statistics
    stats = {}
    for snr in SNRS:
        a = pd.read_csv(CACHE_DIR / f"asr_transcripts_snr{snr}.csv").fillna("")
        stats[snr] = {"ends_?": float(a.asr_raw.str.strip().str.endswith("?").mean()),
                      "ends_!": float(a.asr_raw.str.strip().str.endswith("!").mean()),
                      "empty": float((a.asr_raw.str.strip() == "").mean())}
    (RESULTS_DIR / "noise_asr_punctuation.json").write_text(json.dumps(stats, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    colors = ["#0072B2", "#56B4E9", "#7F7F7F", "#009E73", "#D55E00", "#E69F00"]
    for ax, proto in zip(axes, ("sd", "si")):
        for (pipe, var, label, *_), c in zip(MODELS, colors):
            d = g[(g.protocol == proto) & (g.model == label)].set_index("snr").reindex(order)
            if d["mean"].notna().any():
                ax.errorbar(range(len(order)), 100 * d["mean"], yerr=100 * d["std"], marker="o", capsize=3,
                            label=label, color=c)
        ax.axhline(100 / 7, ls=":", color="k", lw=1)
        ax.set_xticks(range(len(order)), ["clean"] + [f"{s} dB" for s in SNRS])
        ax.set_title({"sd": "Speaker-dependent", "si": "Speaker-independent"}[proto], fontsize=9)
        ax.set_xlabel("test-time SNR")
    axes[0].set_ylabel("accuracy (%)")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Robustness to additive white noise (trained on clean speech)", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "noise_robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
