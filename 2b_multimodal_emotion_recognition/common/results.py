"""Collect per-run outputs into tables."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import EMOTIONS, RUNS_DIR


def collect(pipeline: str | None = None, variant: str | None = None) -> pd.DataFrame:
    rows = []
    for f in sorted(RUNS_DIR.glob("*/*/*/test_metrics.json")):
        run, var, pipe = f.parent.name, f.parent.parent.name, f.parent.parent.parent.name
        if (pipeline and pipe != pipeline) or (variant and var != variant):
            continue
        m = json.loads(f.read_text())
        proto, fold, seed = run.split("_")
        tr = json.loads((f.parent / "train_summary.json").read_text())
        rows.append(dict(pipeline=pipe, variant=var, protocol=proto, fold=int(fold[4:]), seed=int(seed[4:]),
                         accuracy=m["accuracy"], macro_f1=m["macro_f1"], val_acc=tr["best_val_acc"],
                         gate=m.get("mean_gate_speech", np.nan),
                         **{f"f1_{e}": m["per_class_f1"][e] for e in EMOTIONS}))
    return pd.DataFrame(rows)


def summarise(pipeline: str, variant: str, mode: str = "test"):
    df = collect(pipeline, variant)
    if df.empty:
        print("(no evaluated runs yet - run test.py)")
        return df
    g = df.groupby("protocol")[["accuracy", "macro_f1"]].agg(["mean", "std", "count"])
    print(f"\n== {pipeline}/{variant} (test set, mean +- std over runs) ==")
    print((g * [100, 100, 1, 100, 100, 1]).round(2).to_string())
    return df
