"""Cell-by-cell check of every hand-typed table in the 2b report against the file it comes from.

Each table row of report.tex is located by its label, its numbers are parsed in order, and compared with
the values recomputed from Results/ (or runs/) at the precision printed in the report.

    python tests/report_tables_check.py        (exit code 1 on any mismatch)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "Results"
TEX = (ROOT / "report" / "report.tex").read_text(encoding="utf-8").splitlines()
FAILS, CHECKED = [], [0]


def row(label: str, nth: int = 0) -> list[float]:
    """Numbers in the table row whose text (after LaTeX cleanup) starts with `label`."""
    hits = []
    for line in TEX:
        if "&" not in line:
            continue
        clean = re.sub(r"\\(textbf|best|emph)\{", "", line)
        clean = clean.replace("$-$", "-").replace("$", "").replace("\\pm", " ").replace("{", "").replace("}", "")
        first = clean.split("&")[0].strip() + " & " + clean.split("&")[1].strip() if clean.count("&") > 1 else ""
        if clean.strip().startswith(label) or first.startswith(label) or clean.split("&", 1)[1].strip().startswith(label):
            cells = clean.split("&")
            body = " ".join(cells[1:]) if clean.strip().startswith(label) else " ".join(cells[2:])
            body = re.sub(r"\\ref[^ ]*", " ", body).replace("\\to", " ").replace("\\dagger", " ")
            hits.append([float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", body.replace("\\\\", " "))])
    if len(hits) <= nth:
        raise KeyError(label)
    return hits[nth]


def check(table: str, label: str, got: list[float], want: list[float], decimals: list[int] | int):
    if isinstance(decimals, int):
        decimals = [decimals] * len(want)
    if len(got) != len(want):
        FAILS.append(f"{table} | {label}: {len(got)} numbers in report, expected {len(want)}: {got} vs {want}")
        return
    for g, w, d in zip(got, want, decimals):
        CHECKED[0] += 1
        if abs(g - w) > 0.5 * 10 ** -d + 1e-6:      # g must be a correct rounding of w (ties allowed)
            FAILS.append(f"{table} | {label}: report {g} vs data {w:.{d + 2}f}")


def pm(s: str) -> list[float]:
    return [float(x) for x in s.split(" ± ")]


def main():
    acc = pd.read_csv(R / "accuracy_table.csv").set_index("Model")
    main_rows = {
        "Speech & log-Mel": "Speech: log-Mel + BiGRU (baseline)",
        "WavLM + BiGRU (main)": "Speech: WavLM + BiGRU",
        "WavLM + BiGRU + speaker norm": "Speech: WavLM + BiGRU + speaker norm (†)",
        "DistilRoBERTa + BiGRU, ground truth": "Text: DistilRoBERTa + BiGRU (GT transcript)",
        "DistilRoBERTa + BiGRU, Whisper ASR": "Text: DistilRoBERTa + BiGRU (Whisper transcript)",
        "zero-shot emotion model, ground truth": "Text zero-shot emotion-DistilRoBERTa (GT), no training",
        "zero-shot emotion model, Whisper ASR": "Text zero-shot emotion-DistilRoBERTa (Whisper), no training",
        "Gated GMU, ground truth (main)": "Fusion: gated GMU (GT)",
        "Gated GMU, Whisper ASR": "Fusion: gated GMU (Whisper)",
        "Concatenation, ground truth": "Fusion: concat (GT)",
        "Cross-attention, ground truth": "Fusion: cross-attention (GT)",
        "Gated GMU, ground truth + speaker norm": "Fusion: gated GMU (GT) + speaker norm (†)",
        "Gated GMU, Whisper + speaker norm": "Fusion: gated GMU (Whisper) + speaker norm (†)",
    }
    for lab, model in main_rows.items():
        want = sum((pm(acc.loc[model, f"{p} Acc"]) for p in ("RAND", "SD", "SI")), [])
        check("Table main", lab, row(lab), want, 2)

    sig = pd.read_csv(R / "significance.csv").query("protocol == 'si'").set_index(["A", "B"])
    for lab, key in (("speech log-Mel & speech WavLM", ("speech/logmel", "speech/wavlm")),
                     ("speech WavLM & fusion GMU (GT)", ("speech/wavlm", "fusion/gated_wavlm_gt")),
                     ("speech WavLM & fusion GMU (Whisper)", ("speech/wavlm", "fusion/gated_wavlm_asr")),
                     ("speech WavLM & fusion concat", ("speech/wavlm", "fusion/concat_wavlm_gt")),
                     ("speech WavLM & fusion cross-attn", ("speech/wavlm", "fusion/xattn_wavlm_gt")),
                     ("speech WavLM & speech + speaker norm", ("speech/wavlm", "speech/wavlm_spknorm")),
                     ("fusion GMU (GT) & fusion GMU (Whisper)", ("fusion/gated_wavlm_gt", "fusion/gated_wavlm_asr"))):
        r = sig.loc[key]
        got = row(lab)[:5]
        check("Table sig", lab, got, [r.delta_acc_pp, r.ci95_pp, r.p_paired_t, r.B_fixes, r.B_breaks],
              [2, 2, 2 if r.p_paired_t >= 0.1 else 3, 0, 0])

    pros = pd.read_csv(R / "prosody_baseline.csv")
    for lab, model, norm, nth in (("duration only", "duration only (LogReg)", False, 0),
                                  ("44 functionals, LogReg", "44 functionals (LogReg)", False, 0),
                                  ("44 functionals, RBF-SVM", "44 functionals (RBF-SVM)", False, 0),
                                  ("44 functionals, LogReg", "44 functionals (LogReg)", True, 1),
                                  ("44 functionals, RBF-SVM", "44 functionals (RBF-SVM)", True, 1)):
        p = pros[(pros.model == model) & (pros.speaker_norm == norm)].set_index("protocol")
        want = sum(([p.loc[x, "acc_mean"], p.loc[x, "acc_std"]] for x in ("rand", "sd", "si")), [])
        check("Table prosody", f"{lab} norm={norm}", row(lab, nth), want, 2)

    probe = pd.read_csv(R / "layer_probe.csv")
    w, lm = probe[probe.store == "wavlm"], probe[probe.store == "logmel"].iloc[0]
    for lab, col in (("emotion, SD", "emotion_sd"), ("emotion, SI &", "emotion_si"),
                     ("emotion, SI + speaker norm", "emotion_si_spknorm"), ("speaker identity", "speaker_id")):
        want = list(100 * w[col]) + [100 * lm[col]]
        got = row(lab.rstrip(" &"), 0 if lab != "emotion, SI &" else 0)
        if lab == "emotion, SI &":
            got = [r for r in (row("emotion, SI", i) for i in range(2)) if len(r) == 8][0]
        check("Table probe", lab, got, want, 1)

    abl = pd.read_csv(R / "ablation_speech.csv").set_index(["variant", "protocol"])
    for lab, var in (("main: 7 layers", "wavlm"), ("all 13 layers", "wavlm_full"),
                     ("layer logits: no weight decay", "wavlm_lw0"), ("layer logits: no decay, 10", "wavlm_lw0x10")):
        sd, si = abl.loc[(var, "sd")], abl.loc[(var, "si")]
        want = [sd.acc_mean, sd.acc_std, si.acc_mean, si.acc_std]
        if var != "wavlm":
            want += [si.delta_pp, si.ci95_pp, si.p_paired_t]
        want += [si.max_layer_weight]
        got = row(lab)
        check("Table ablation", lab, got, want, [2, 2, 2, 2] + ([2, 2, 2] if var != "wavlm" else []) + [3])

    f1 = pd.read_csv(R / "per_class_f1.csv").query("Protocol == 'SI'").set_index("Model")
    for lab, model in (("Speech log-Mel", "Speech: log-Mel + BiGRU (baseline)"), ("Speech WavLM", "Speech: WavLM + BiGRU"),
                       ("Text, Whisper ASR", "Text: DistilRoBERTa + BiGRU (Whisper transcript)"),
                       ("Fusion GMU (GT)", "Fusion: gated GMU (GT)"), ("Fusion cross-attn (GT)", "Fusion: cross-attention (GT)")):
        check("Table F1", lab, row(lab), list(f1.loc[model].values[1:]), 1)

    asr = pd.read_csv(R / "asr_transcript_stats_by_emotion.csv").set_index("emotion")
    order = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
    for lab, col in (("ends with ``!''", "ends_excl"), ("ends with ``.''", "ends_period"),
                     ("ends with ``?''", "ends_q"), ("exact carrier sentence", "exact_carrier")):
        check("Table ASR", lab, row(lab), list(asr.loc[order, col]), 1)

    fg = pd.read_csv(R / "fusion_gain_by_speech_confidence_gated_wavlm_asr_si.csv")
    check("Table bins", "clips", row("clips"), list(fg.n), 0)
    check("Table bins", "speech-only acc.", row("speech-only acc."), list(fg.speech_acc), 1)
    check("Table bins", "fusion (Whisper) acc.", row("fusion (Whisper) acc."), list(fg.fusion_acc), 1)

    sil = pd.read_csv(R / "silhouette_scores.csv")
    for p in ("SD", "SI"):
        line = [l for l in TEX if l.strip().startswith(f"{p} & 0.")][0]
        got = [float(x) for x in re.findall(r"-?\d+\.\d+", line.replace("$-$", "-").replace("$", ""))]
        check("Table silhouette", p, got, list(sil[sil.protocol == p.lower()].silhouette), 3)

    noise = (R / "noise_robustness.md").read_text(encoding="utf-8")
    def nrow(proto, name):
        line = [l for l in noise.splitlines() if f"('{proto}', '{name}')" in l][0]
        return [float(c.split("±")[0]) for c in line.split("|")[2:-1]]
    tex_noise = [l for l in TEX if re.match(r"^(SD|SI)? ?& (Speech|Fusion|Text)", l.strip()) and "dB" not in l]
    for (proto, lab, name) in (("sd", "Speech WavLM", "Speech-only (WavLM)"), ("sd", "Speech + speaker norm", "Speech-only + spk-norm"),
                               ("sd", "Fusion GMU (Whisper) &", "Fusion GMU (Whisper text)"),
                               ("sd", "Fusion GMU (Whisper) + speaker norm", "Fusion GMU (Whisper) + spk-norm"),
                               ("si", "Speech WavLM", "Speech-only (WavLM)"), ("si", "Speech + speaker norm", "Speech-only + spk-norm"),
                               ("si", "Fusion GMU (GT)", "Fusion GMU (GT text)"),
                               ("si", "Fusion GMU (Whisper) + speaker norm", "Fusion GMU (Whisper) + spk-norm"),
                               ("si", "Text only", "Text-only (Whisper)")):
        start = [i for i, l in enumerate(TEX) if l.strip().startswith("SD & Speech WavLM")][0]
        block = TEX[start:start + 10]
        idx0 = 0 if proto == "sd" else 4
        cand = [l for l in block[idx0:] if lab.rstrip(" &") in l and (lab.endswith("&") is False or "speaker" not in l)]
        nums = re.findall(r"\d+\.\d+", cand[0].replace("\\textbf{", ""))
        check("Table noise", f"{proto} {lab}", [float(x) for x in nums], nrow(proto, name), 1)

    si = pd.read_csv(R / "si_linear.csv")
    def run_acc(var, fold):
        return [100 * json.loads((f / "test_metrics.json").read_text())["accuracy"]
                for f in sorted((ROOT / "runs" / "speech" / var).glob(f"si_fold{fold}_seed*"))]
    b0, b1 = run_acc("wavlm", 0), run_acc("wavlm", 1)
    check("Table prereg", "WavLM + BiGRU", row("WavLM + BiGRU (Table"),
          [np.mean(b0), min(b0), max(b0), np.mean(b1), min(b1), max(b1), (np.mean(b0) + np.mean(b1)) / 2], 1)
    l = si[si.model == "all-layer linear"].accuracy.values
    check("Table prereg", "all-layer linear", row("all-layer linear (pre"), [l[0], l[1], l.mean()], 1)
    n0, n1 = run_acc("wavlm_spknorm", 0), run_acc("wavlm_spknorm", 1)
    check("Table prereg", "BiGRU + norm", row("WavLM + BiGRU + speaker norm", 1),
          [np.mean(n0), np.mean(n1), (np.mean(n0) + np.mean(n1)) / 2], 1)
    l = si[si.model.str.startswith("all-layer linear + speaker")].accuracy.values
    check("Table prereg", "all-layer linear + norm", row("all-layer linear + speaker norm"), [l[0], l[1], l.mean()], 1)

    cap = pd.read_csv(R / "capacity_scores.csv")
    for lab, m in (("B1 BiGRU", "B1"), ("B2 BiGRU", "B2"), ("B3 small", "B3"), ("B4 attentive", "B4"),
                   ("L1 logistic", "L1"), ("L1n same", "L1n"), ("L2n mean", "L2n")):
        line = [l for l in TEX if l.strip().startswith(lab)][0]
        body = line.split("&", 2)[2].replace("\textbf{", "").replace("}", "")
        got = [float(x) for x in re.findall(r"\d+\.\d+", body)]
        a, b = cap[(cap.model == m) & (cap.direction == "OAF->YAF")].iloc[0], cap[(cap.model == m) & (cap.direction == "YAF->OAF")].iloc[0]
        if m.startswith("B"):
            want = [a["mean"], a["min"], a["max"], b["mean"], b["min"], b["max"], (a["mean"] + b["mean"]) / 2,
                    (a.ensemble + b.ensemble) / 2]
        else:
            want = [a["mean"], b["mean"], (a["mean"] + b["mean"]) / 2]
        check("Table capacity", lab, got, want, 1)

    aug = pd.read_csv(R / "augmentation_scores.csv")
    for lab, m in (("A-B2 = B2", "A-B2"), ("A-L1n = L1n", "A-L1n")):
        line = [l for l in TEX if l.strip().startswith(lab)][0]
        body = line.split("&", 2)[2].replace("\\textbf{", "").replace("}", "")
        got = [float(x) for x in re.findall(r"\d+\.\d+", body)]
        a = aug[(aug.model == m) & (aug.direction == "OAF->YAF")].iloc[0]
        b = aug[(aug.model == m) & (aug.direction == "YAF->OAF")].iloc[0]
        want = ([a["mean"], a["min"], a["max"], b["mean"], b["min"], b["max"], (a["mean"] + b["mean"]) / 2,
                 (a.ensemble + b.ensemble) / 2] if m == "A-B2" else [a["mean"], b["mean"], (a["mean"] + b["mean"]) / 2])
        check("Table capacity", lab, got, want, 1)

    bb = pd.read_csv(R / "backbone_scores.csv").set_index(["model", "direction"]).acc
    bt = pd.read_csv(R / "backbone_tests.csv").set_index(["comparison", "direction"])
    dirs = ("OAF->YAF", "YAF->OAF")
    for lab, m in (("B: WavLM-Base+", "A-L1n(B)"), ("WL: WavLM-Large", "A-L1n(WL)"), ("HL: HuBERT-Large", "A-L1n(HL)"),
                   ("E: emotion2vec base", "A-L1n(E)"), ("ENS (primary", "ENS"), ("LC: emotion2vec+ seed, our", "A-L1n(LC)"),
                   ("LC: its own head", "LC-a zero-shot (emotion2vec+ seed head)")):
        line = [l for l in TEX if l.strip().startswith(lab)][0]
        body = line.split("&", 1)[1].replace("\textbf{", "").replace("}", "").replace("$-$", "-")
        got = [float(x) for x in re.findall(r"-?\d+\.\d+", body)]
        want = [bb[(m, d)] for d in dirs] + [(bb[(m, dirs[0])] + bb[(m, dirs[1])]) / 2]
        if m not in ("A-L1n(B)",) and not m.startswith(("A-L1n(LC", "LC")):
            for d in dirs:
                r = bt.loc[(f"{m} - A-L1n(B)", d)]
                want += [r.delta_pp, r.ci_lo, r.ci_hi]
        check("Table backbones", lab, got, want, 1)

    fs = pd.read_csv(R / "final_scores.csv").set_index(["model", "direction"]).acc
    ft = pd.read_csv(R / "final_tests.csv").set_index(["comparison", "direction"])
    for lab, m in (("R: A-L1n, WavLM-Large", "A-L1n(WL)"), ("W3: A-L1n, Whisper", "A-L1n(W3)"),
                   ("P: R + pitch", "AP-L1n(WL)"), ("S: R + self-training", "ST(A-L1n(WL))"),
                   ("ENS2: P on both", "ENS2"), ("FINAL (primary)", "FINAL")):
        line = [l for l in TEX if l.strip().startswith(lab)][0]
        body = line.split("&", 1)[1].replace("\textbf{", "").replace("}", "").replace("$-$", "-")
        got = [float(x) for x in re.findall(r"-?\d+\.\d+", body)]
        want = [fs[(m, d)] for d in dirs] + [(fs[(m, dirs[0])] + fs[(m, dirs[1])]) / 2]
        if m not in ("A-L1n(WL)", "ENS2"):
            for d in dirs:
                r = ft.loc[(f"{m} - A-L1n(WL)", d)]
                want += [r.delta_pp, r.ci_lo, r.ci_hi]
        check("Table final", lab, got, want, 1)

    print(f"{CHECKED[0]} table cells checked, {len(FAILS)} mismatches")
    for f in FAILS:
        print("  MISMATCH", f)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
