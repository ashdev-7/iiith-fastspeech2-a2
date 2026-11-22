"""Reproduce every number in Results/ with one command.

    python run_all.py                # prepare data (if needed), train + test all variants, all analyses
    python run_all.py --skip-prepare # features already cached
    python run_all.py --ablations    # also the speech-feature / layer-weighting ablation (+5.6 GB cache)
    python run_all.py --skip-prepare --skip-train --skip-analyses --ablations   # only the ablation

Each variant is trained/tested under all three protocols (rand, sd, si) unless noted.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

EXPERIMENTS = [
    ("speech_pipeline", ["--features", "wavlm"]),
    ("speech_pipeline", ["--features", "logmel"]),
    ("speech_pipeline", ["--features", "wavlm", "--spk_norm", "true"]),
    ("text_pipeline", ["--transcripts", "gt"]),
    ("text_pipeline", ["--transcripts", "asr"]),
    ("fusion_pipeline", ["--fusion", "gated", "--transcripts", "gt"]),
    ("fusion_pipeline", ["--fusion", "gated", "--transcripts", "asr"]),
    ("fusion_pipeline", ["--fusion", "concat", "--transcripts", "gt"]),
    ("fusion_pipeline", ["--fusion", "xattn", "--transcripts", "gt"]),
    ("fusion_pipeline", ["--fusion", "gated", "--transcripts", "gt", "--spk_norm", "true"]),
    ("fusion_pipeline", ["--fusion", "gated", "--transcripts", "asr", "--spk_norm", "true"]),
]
# Ablation of two design assumptions of the speech front-end (Report, section A):
#  wavlm_full   : all 13 WavLM layers at native 20 ms instead of 7 layers at 40 ms
#  wavlm_lw0    : no weight decay on the 7 softmax layer-weight logits
#  wavlm_lw0x10 : no weight decay and 10x learning rate on those logits
ABLATIONS = [
    ("speech_pipeline", ["--features", "wavlm_full"]),
    ("speech_pipeline", ["--features", "wavlm", "--layer_wd", "0"]),
    ("speech_pipeline", ["--features", "wavlm", "--layer_wd", "0", "--layer_lr_mult", "10"]),
]
ANALYSES = ["analysis/layer_probe.py", "analysis/prosody_baseline.py", "analysis/noise_robustness.py",
            "analysis/make_results.py", "analysis/error_analysis.py",
            "analysis/report_figures.py"]  # the report's figures, at print size, from the outputs above


def sh(*args, check=True):
    t = time.time()
    print("\n$", " ".join(map(str, args)), flush=True)
    rc = subprocess.call([PY, *map(str, args)], cwd=ROOT)
    print(f"  -> exit {rc} ({time.time() - t:.0f}s)", flush=True)
    if check and rc:
        raise SystemExit(rc)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-analyses", action="store_true")
    ap.add_argument("--ablations", action="store_true")
    args = ap.parse_args()
    if not args.skip_prepare:
        sh("prepare_data.py", "--all")
    if not args.skip_train:
        for pipe, extra in EXPERIMENTS:
            sh(f"models/{pipe}/train.py", *extra)
            sh(f"models/{pipe}/test.py", *extra)
    failed = [] if args.skip_analyses else [a for a in ANALYSES if sh(a, check=False)]  # independent: keep going
    if args.ablations:
        sh("prepare_data.py", "--speech_full")
        sh("analysis/check_full_store.py")
        for pipe, extra in ABLATIONS:
            sh(f"models/{pipe}/train.py", *extra)
            sh(f"models/{pipe}/test.py", *extra)
        failed += [a for a in ["analysis/ablation_speech.py"] if sh(a, check=False)]
    if failed:
        print("FAILED analyses:", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
