"""POST-HOC (not pre-registered) sensitivity of the pre-registered linear model to its one hyper-parameter C.

The pre-registered training-speaker CV could not discriminate between C values (100 % for all), so C = 0.001
was chosen by tie-break, at the edge of the grid. This script shows how SI accuracy depends on C over a wider
range, so a reader can see whether 66.0 / 70.6 % is a lucky pick. It reads the test speaker for every C, so it
is a diagnostic only: no value here replaces the pre-registered numbers.

    python analysis/c_sensitivity.py   ->  Results/c_sensitivity.csv, Results/c_sensitivity.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.layer_probe import pooled, spk_norm  # noqa: E402
from analysis.si_linear import clf  # noqa: E402
from common.config import RESULTS_DIR, SPEAKERS  # noqa: E402
from common.data import load_index, make_split  # noqa: E402

C_WIDE = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 10.0)


def main():
    df = load_index()
    y, spk = df.label.values, df.speaker.values
    X = pooled("wavlm").reshape(len(df), -1)
    feats = {"L1 (no norm)": X, "L1n (speaker norm †)": spk_norm(X, spk)}
    trte = [(np.concatenate([s["train"], s["val"]]), s["test"])
            for s in (make_split(df, "si", 0, f) for f in (0, 1))]
    rows = []
    for name, F in feats.items():
        for C in C_WIDE:
            acc = []
            for fold, (tr, te) in enumerate(trte):
                acc.append(100 * (clf(C).fit(F[tr], y[tr]).predict(F[te]) == y[te]).mean())
                rows.append(dict(model=name, C=C, direction=f"{SPEAKERS[fold]}->{SPEAKERS[1 - fold]}",
                                 accuracy=round(acc[-1], 2)))
            print(name, C, [round(a, 2) for a in acc], round(np.mean(acc), 2), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS_DIR / "c_sensitivity.csv", index=False)
    w = t.pivot_table(index=["model", "C"], columns="direction", values="accuracy").reset_index()
    w["mean"] = w[[c for c in w.columns if "->" in c]].mean(axis=1).round(2)
    (RESULTS_DIR / "c_sensitivity.md").write_text(
        "# POST-HOC sensitivity of the pre-registered linear model to C\n\n"
        "Not pre-registered: every row reads the test speaker. The pre-registered value is C = 0.001 "
        "(chosen by tie-break; training-speaker CV was 100 % for every C). Diagnostic only.\n\n"
        + w.to_markdown(index=False) + "\n", encoding="utf-8")
    print(w.to_string(index=False))


if __name__ == "__main__":
    main()
