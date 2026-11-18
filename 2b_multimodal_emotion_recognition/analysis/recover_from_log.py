"""Rebuild the outputs of layer_probe / prosody_baseline / noise_robustness from a run log.

Why this exists: in the Kaggle run (logs/kaggle_run_v2.log) these three analyses computed
all their numbers, printed them per run, and then crashed when saving because Results/ did
not exist yet (fixed since in common/config.py). The checkpoints needed to recompute the
noise study were deleted after the run, so instead of retraining 176 models we parse the
printed per-run values and feed them to the *same* report() functions the scripts use.

Precision: layer-probe and noise accuracies were printed with 3 decimals (0.1 pp), the
prosody baseline at full precision. Re-running the three scripts reproduces them exactly.

    python analysis/recover_from_log.py [logs/kaggle_run_v2.log]
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import pandas as pd  # noqa: E402


def section(text: str, script: str) -> list[str]:
    start = text.index(f"$ analysis/{script}")
    end = text.find("\n$ ", start + 1)
    return text[start:end if end > 0 else None].splitlines()


def dict_rows(lines: list[str], key: str) -> list[dict]:
    rows = []
    for ln in lines:
        if ln.startswith("{'" + key):
            rows.append(ast.literal_eval(re.sub(r"np\.float64\(([^)]*)\)", r"\1", ln)))
    return rows


def main():
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "kaggle_run_v2.log"
    text = log.read_text(encoding="utf-8")

    import layer_probe
    t = pd.DataFrame(dict_rows(section(text, "layer_probe.py"), "store"))
    layer_probe.report(t)
    print(f"layer_probe: {len(t)} rows")

    import prosody_baseline
    t = pd.DataFrame(dict_rows(section(text, "prosody_baseline.py"), "model"))
    prosody_baseline.report(t)
    print(f"prosody_baseline: {len(t)} rows")

    import noise_robustness as nr
    nr.CACHE_DIR = ROOT / "artifacts"  # noisy-ASR transcripts exported from the Kaggle cache
    labels = {label: (pipe, var) for pipe, var, label, *_ in nr.MODELS}
    pat = re.compile(r"^(?P<label>.+?) (?P<run>(?:sd|si|rand)_fold\d+_seed\d+) (?P<accs>\[.*\])$")
    rows = []
    for ln in section(text, "noise_robustness.py"):
        m = pat.match(ln)
        if not m or m["label"] not in labels:
            continue
        accs = ast.literal_eval(m["accs"])
        assert len(accs) == 1 + len(nr.SNRS), ln
        pipe, var = labels[m["label"]]
        for snr, a in zip(["clean"] + nr.SNRS, accs):
            rows.append(dict(model=m["label"], pipeline=pipe, variant=var, protocol=m["run"].split("_")[0],
                             run=m["run"], snr=snr, acc=float(a)))
    t = pd.DataFrame(rows)
    n_runs = t.groupby("model").run.nunique()
    print("noise_robustness runs per model:\n" + n_runs.to_string())
    nr.report(t)


if __name__ == "__main__":
    main()
