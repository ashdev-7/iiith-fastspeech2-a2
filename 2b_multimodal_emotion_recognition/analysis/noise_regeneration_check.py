"""Pre-registration 3, part N (analysis/PREREGISTRATION_si_augmentation.md): checks N1 and N2.

N1: every retrained run of the six noise-study models (Kaggle job 'noise') is compared with the committed
    run in runs/: share of identical test predictions and accuracy difference. The ASR transcripts
    re-generated in that job are compared with artifacts/asr_transcripts.csv.
N2: the regenerated per-run noise results are compared cell by cell with the log-recovered ones.

    python analysis/noise_regeneration_check.py --job_dir <downloaded kaggle output>/mer
        [--old_noise_csv <copy of the log-recovered Results/noise_robustness_runs.csv>]
    -> Results/noise_regeneration_check.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from common.config import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

MODELS = [("speech", "wavlm"), ("speech", "wavlm_spknorm"), ("text", "asr"), ("fusion", "gated_wavlm_gt"),
          ("fusion", "gated_wavlm_asr"), ("fusion", "gated_wavlm_asr_spknorm")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_dir", required=True)
    ap.add_argument("--old_noise_csv", default=str(RESULTS_DIR / "noise_robustness_runs.csv"))
    args = ap.parse_args()
    job = Path(args.job_dir)
    out = ["# Regeneration of the noise study: checks N1 and N2\n",
           "Plan: `analysis/PREREGISTRATION_si_augmentation.md`, part N.\n"]

    # N1a: transcripts
    new_asr = job / "cache_artifacts" / "asr_transcripts.csv"
    if new_asr.exists():
        a = pd.read_csv(PROJECT_ROOT / "artifacts" / "asr_transcripts.csv").fillna("")
        b = pd.read_csv(new_asr).fillna("")
        assert (a.uid.values == b.uid.values).all()
        same = (a.asr_raw.values == b.asr_raw.values)
        out.append(f"## N1a: Whisper transcripts\n\n{same.sum()} of {len(same)} clean transcripts identical "
                   f"to `artifacts/asr_transcripts.csv` ({100 * same.mean():.2f} %).\n")
        if (~same).any():
            d = pd.DataFrame({"uid": a.uid[~same], "committed": a.asr_raw[~same], "regenerated": b.asr_raw[~same]})
            out.append(d.head(30).to_markdown(index=False) + "\n")

    # N1b: predictions
    rows = []
    for pipe, var in MODELS:
        for new in sorted((job / "runs" / pipe / var).glob("*/test_predictions.csv")):
            old = PROJECT_ROOT / "runs" / pipe / var / new.parent.name / "test_predictions.csv"
            if not old.exists():
                continue
            o, n = pd.read_csv(old).set_index("uid"), pd.read_csv(new).set_index("uid")
            n = n.loc[o.index]
            rows.append(dict(model=f"{pipe}/{var}", run=new.parent.name,
                             acc_committed=round(100 * (o.pred == o.true).mean(), 2),
                             acc_retrained=round(100 * (n.pred == n.true).mean(), 2),
                             identical_pred_pct=round(100 * (o.pred == n.pred).mean(), 2)))
    r = pd.DataFrame(rows)
    out.append(f"## N1b: retrained vs. committed runs ({len(r)} runs)\n\n"
               f"Runs with 100 % identical predictions: {(r.identical_pred_pct == 100).sum()} of {len(r)}.\n\n"
               + r.to_markdown(index=False) + "\n")

    # N2: regenerated vs. log-recovered noise results
    new_noise = pd.read_csv(job / "Results" / "noise_robustness_runs.csv")
    old_noise = pd.read_csv(args.old_noise_csv)
    key = ["model", "protocol", "run", "snr"]
    for t in (new_noise, old_noise):
        t["snr"] = t["snr"].astype(str)
    m = old_noise.merge(new_noise, on=key, how="outer", suffixes=("_recovered", "_regenerated"), indicator=True)
    m["diff_pp"] = (100 * (m.acc_regenerated - m.acc_recovered)).round(2)
    big = m[(m.diff_pp.abs() > 0.1) | (m._merge != "both")]
    out.append(f"## N2: regenerated vs. log-recovered per-run noise accuracies\n\n{len(m)} cells; "
               f"{(m._merge == 'both').sum()} matched; max |difference| {m.diff_pp.abs().max():.2f} pp; "
               f"{len(big)} cells differ by more than 0.1 pp or are unmatched.\n\n"
               + (big[key + ["acc_recovered", "acc_regenerated", "diff_pp", "_merge"]].to_markdown(index=False)
                  if len(big) else "") + "\n")
    (RESULTS_DIR / "noise_regeneration_check.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
