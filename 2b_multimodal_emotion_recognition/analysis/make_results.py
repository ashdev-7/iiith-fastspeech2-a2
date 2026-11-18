"""Aggregate all runs into the Results/ folder: accuracy tables, confusion
matrices, representation-separability plots, fusion analysis and error analysis.

    python analysis/make_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from common.config import (cached, CACHE_DIR, EMOTION_NAMES, EMOTIONS, PLOTS_DIR, RESULTS_DIR,  # noqa: E402
                           RUNS_DIR, TEXT_PLM_LABEL_MAP)
from common.data import load_index, make_split  # noqa: E402
from common.results import collect  # noqa: E402

# colour-blind-safe categorical palette (Okabe-Ito), one fixed colour per emotion
PALETTE = dict(zip(EMOTIONS, ["#D55E00", "#009E73", "#7F3C8D", "#E69F00", "#7F7F7F", "#56B4E9", "#0072B2"]))
PROTO_NAME = {"rand": "Random clip-level split", "sd": "Speaker-dependent (word-disjoint)",
              "si": "Speaker-independent (LOSO)"}
PROTOS = ("rand", "sd", "si")

# rows of the main table: (pipeline, variant, display name)
MAIN = [
    ("speech", "logmel", "Speech: log-Mel + BiGRU (baseline)"),
    ("speech", "wavlm", "Speech: WavLM + BiGRU"),
    ("speech", "wavlm_spknorm", "Speech: WavLM + BiGRU + speaker norm (†)"),
    ("text", "gt", "Text: DistilRoBERTa + BiGRU (GT transcript)"),
    ("text", "asr", "Text: DistilRoBERTa + BiGRU (Whisper transcript)"),
    ("fusion", "concat_wavlm_gt", "Fusion: concat (GT)"),
    ("fusion", "xattn_wavlm_gt", "Fusion: cross-attention (GT)"),
    ("fusion", "gated_wavlm_gt", "Fusion: gated GMU (GT)"),
    ("fusion", "gated_wavlm_asr", "Fusion: gated GMU (Whisper)"),
    ("fusion", "gated_wavlm_gt_spknorm", "Fusion: gated GMU (GT) + speaker norm (†)"),
    ("fusion", "gated_wavlm_asr_spknorm", "Fusion: gated GMU (Whisper) + speaker norm (†)"),
]


def pm(s: pd.Series) -> str:
    return f"{100 * s.mean():.2f} ± {100 * s.std(ddof=1 if len(s) > 1 else 0):.2f}" if len(s) else "-"


# ------------------------------------------------------------------ tables
def zero_shot_rows(df_idx):
    rows = []
    for name in ("text_gt", "text_asr"):
        f = cached(f"{name}.zeroshot_probs.npy")
        if not f.exists():
            continue
        probs = np.load(f)
        labels = cached(f"{name}.zeroshot_labels.txt").read_text().split("\n")
        mapped = np.array([EMOTIONS.index(TEXT_PLM_LABEL_MAP[labels[i]]) for i in probs.argmax(1)])
        from common.config import PROTOCOLS
        for proto, runs in ((k, v["runs"]) for k, v in PROTOCOLS.items()):
            accs = []
            for seed, fold in runs:
                te = make_split(df_idx, proto, seed, fold)["test"]
                accs.append((mapped[te] == df_idx.label.values[te]).mean())
            rows.append(dict(model=f"Text zero-shot emotion-DistilRoBERTa ({'GT' if 'gt' in name else 'Whisper'}), no training",
                             protocol=proto, acc=pd.Series(accs)))
        pred_dist = pd.Series([EMOTIONS[i] for i in mapped]).value_counts(normalize=True).round(3).to_dict()
        (RESULTS_DIR / f"zeroshot_{name}_prediction_distribution.json").write_text(json.dumps(pred_dist, indent=2))
    return rows


def accuracy_tables(runs: pd.DataFrame, df_idx):
    lines = ["# Accuracy tables", "",
             "Test-set accuracy and macro-F1 (%), mean ± std over runs. ",
             "**RAND** = clip-level random stratified 70/15/15 split as used in most TESS papers (optimistic: same speakers and words in train and test; 5 seeds). ",
             "**SD** = speaker-dependent, word-disjoint split (5 seeds; 1960/420/420 clips). ",
             "**SI** = speaker-independent leave-one-speaker-out (2 folds × 3 seeds; train 1190 / val 210 / test 1400 clips of the unseen speaker). ",
             "Chance = 14.29 % (7 balanced classes). (†) speaker normalisation uses label-free per-speaker feature statistics, including the unlabelled audio of the test speaker under SI (transductive).", ""]
    table = []
    for pipe, var, name in MAIN:
        d = runs[(runs.pipeline == pipe) & (runs.variant == var)]
        if d.empty:
            continue
        r = {"Model": name}
        for proto in PROTOS:
            p = d[d.protocol == proto]
            r[f"{proto.upper()} Acc"] = pm(p.accuracy)
            r[f"{proto.upper()} Macro-F1"] = pm(p.macro_f1)
            r[f"{proto.upper()} runs"] = len(p)
        table.append(r)
    for z in zero_shot_rows(df_idx):
        r = next((t for t in table if t["Model"] == z["model"]), None)
        if r is None:
            r = {"Model": z["model"]}
            table.append(r)
        r[f"{z['protocol'].upper()} Acc"] = pm(z["acc"])
    table.append({"Model": "Chance (uniform)", "RAND Acc": "14.29", "SD Acc": "14.29", "SI Acc": "14.29"})
    t = pd.DataFrame(table).fillna("-")
    t.to_csv(RESULTS_DIR / "accuracy_table.csv", index=False)
    lines += [t.to_markdown(index=False), ""]

    # per-direction SI table
    lines += ["## Speaker-independent accuracy by direction (%)", ""]
    rows = []
    for pipe, var, name in MAIN:
        d = runs[(runs.pipeline == pipe) & (runs.variant == var) & (runs.protocol == "si")]
        if d.empty:
            continue
        rows.append({"Model": name, "train OAF → test YAF": pm(d[d.fold == 0].accuracy),
                     "train YAF → test OAF": pm(d[d.fold == 1].accuracy)})
    t2 = pd.DataFrame(rows)
    t2.to_csv(RESULTS_DIR / "accuracy_si_by_direction.csv", index=False)
    lines += [t2.to_markdown(index=False), ""]

    # per-class F1
    lines += ["## Per-class F1 (%), mean over runs", ""]
    rows = []
    for pipe, var, name in MAIN:
        for proto in PROTOS:
            d = runs[(runs.pipeline == pipe) & (runs.variant == var) & (runs.protocol == proto)]
            if d.empty:
                continue
            rows.append({"Model": name, "Protocol": proto.upper(),
                         **{EMOTION_NAMES[e]: f"{100 * d[f'f1_{e}'].mean():.1f}" for e in EMOTIONS}})
    t3 = pd.DataFrame(rows)
    t3.to_csv(RESULTS_DIR / "per_class_f1.csv", index=False)
    lines += [t3.to_markdown(index=False), ""]

    # all individual runs
    runs.sort_values(["pipeline", "variant", "protocol", "fold", "seed"]).round(4).to_csv(
        RESULTS_DIR / "all_runs.csv", index=False)
    (RESULTS_DIR / "accuracy_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# ------------------------------------------------------------------ confusion matrices
def confusion_plots(runs):
    for pipe, var, name in MAIN:
        for proto in PROTOS:
            mats = []
            for f in (RUNS_DIR / pipe / var).glob(f"{proto}_*/test_metrics.json"):
                mats.append(np.array(json.loads(f.read_text())["confusion"]))
            if not mats:
                continue
            cm = np.sum(mats, 0)
            cmn = cm / cm.sum(1, keepdims=True)
            fig, ax = plt.subplots(figsize=(5.2, 4.4))
            im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for i in range(len(EMOTIONS)):
                for j in range(len(EMOTIONS)):
                    ax.text(j, i, f"{100 * cmn[i, j]:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if cmn[i, j] > 0.6 else "black")
            ticks = [EMOTION_NAMES[e].replace("Pleasant surprise", "Pl. surprise") for e in EMOTIONS]
            ax.set_xticks(range(7), ticks, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(7), ticks, fontsize=8)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{name}\n{PROTO_NAME[proto]} (row-normalised %, {len(mats)} runs)", fontsize=8)
            fig.colorbar(im, fraction=0.046)
            fig.savefig(PLOTS_DIR / f"confusion_{pipe}_{var}_{proto}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


# ------------------------------------------------------------------ representation separability
def tsne_panel(ax, X, y, spk, title):
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
    sil = silhouette_score(X, y, metric="cosine") if len(np.unique(y)) > 1 else np.nan
    for e_i, e in enumerate(EMOTIONS):
        for s, mk in (("OAF", "o"), ("YAF", "^")):
            m = (y == e_i) & (spk == s)
            ax.scatter(Z[m, 0], Z[m, 1], s=7, c=PALETTE[e], marker=mk, alpha=0.75, linewidths=0,
                       label=EMOTION_NAMES[e] if s == "OAF" else None)
    ax.set_title(f"{title}\nsilhouette (cosine) = {sil:.3f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return sil


def separability(df_idx):
    """t-SNE of test-set embeddings from the temporal (speech), contextual (text)
    and fusion blocks. Silhouette scores are computed in the original space."""
    rows = []
    for proto, run in (("sd", "sd_fold0_seed0"), ("si", "si_fold0_seed0")):
        panels = [
            ("speech", "wavlm", "temporal", "Temporal block (speech-only model)"),
            ("speech", "wavlm_spknorm", "temporal", "Temporal block (speech + spk-norm)"),
            ("text", "gt", "contextual", "Contextual block (text-only, GT)"),
            ("text", "asr", "contextual", "Contextual block (text-only, Whisper)"),
            ("fusion", "gated_wavlm_gt", "fusion", "Fusion block (gated, GT)"),
            ("fusion", "gated_wavlm_asr", "fusion", "Fusion block (gated, Whisper)"),
        ]
        panels = [p for p in panels if (RUNS_DIR / p[0] / p[1] / run / "test_embeddings.npz").exists()]
        if not panels:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
        axes = np.atleast_1d(axes)
        for ax, (pipe, var, key, title) in zip(axes, panels):
            z = np.load(RUNS_DIR / pipe / var / run / "test_embeddings.npz")
            spk = df_idx.speaker.values[z["idx"]]
            sil = tsne_panel(ax, z[key], z["label"], spk, title)
            rows.append(dict(protocol=proto, run=run, block=title, silhouette=sil))
        axes[0].legend(fontsize=7, loc="lower left", markerscale=2, frameon=True)
        fig.suptitle(f"Test-set representations, {PROTO_NAME[proto]} ({run}); ● OAF  ▲ YAF", fontsize=10)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"tsne_blocks_{proto}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(rows).round(4).to_csv(RESULTS_DIR / "silhouette_scores.csv", index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ fusion analysis
def paired_predictions(pipe_a, var_a, pipe_b, var_b, proto):
    out = []
    for f in sorted((RUNS_DIR / pipe_a / var_a).glob(f"{proto}_*/test_predictions.csv")):
        g = RUNS_DIR / pipe_b / var_b / f.parent.name / "test_predictions.csv"
        if not g.exists():
            continue
        a, b = pd.read_csv(f), pd.read_csv(g)
        assert (a.uid.values == b.uid.values).all()
        pa = a[[f"p_{e}" for e in EMOTIONS]].values
        m = pd.DataFrame({"run": f.parent.name, "uid": a.uid, "speaker": a.speaker, "true": a.true,
                          "speech_pred": a.pred, "speech_conf": pa.max(1), "fusion_pred": b.pred})
        if "gate_speech" in b:
            m["gate_speech"] = b.gate_speech
        out.append(m)
    return pd.concat(out) if out else pd.DataFrame()


def fusion_analysis(runs):
    lines = ["# When does fusion help?", ""]
    # 1) paired gains per protocol / transcript type
    rows = []
    for var in ("gated_wavlm_gt", "gated_wavlm_asr", "concat_wavlm_gt", "xattn_wavlm_gt"):
        for proto in ("sd", "si"):
            m = paired_predictions("speech", "wavlm", "fusion", var, proto)
            if m.empty:
                continue
            sc, fc = (m.speech_pred == m.true), (m.fusion_pred == m.true)
            rows.append(dict(fusion=var, protocol=proto, speech_acc=100 * sc.mean(), fusion_acc=100 * fc.mean(),
                             delta=100 * (fc.mean() - sc.mean()), fixed=int((~sc & fc).sum()),
                             broken=int((sc & ~fc).sum()),
                             mean_gate_speech=m.gate_speech.mean() if "gate_speech" in m else np.nan))
    t = pd.DataFrame(rows)
    t.round(3).to_csv(RESULTS_DIR / "fusion_paired_gains.csv", index=False)
    lines += ["## Paired comparison against the speech-only model (same split & seed)", "",
              "`fixed` = clips the speech model got wrong and fusion got right; `broken` = the reverse. "
              "`mean_gate_speech` = mean GMU gate z (1 = rely fully on speech).", "",
              t.round(2).to_markdown(index=False), ""]

    # 2) gain as a function of speech-model confidence
    for var in ("gated_wavlm_asr", "gated_wavlm_gt"):
        m = paired_predictions("speech", "wavlm", "fusion", var, "si")
        if m.empty:
            continue
        bins = [0, 0.5, 0.7, 0.9, 0.99, 1.0001]
        m["conf_bin"] = pd.cut(m.speech_conf, bins, right=False)
        g = m.groupby("conf_bin", observed=True).apply(
            lambda d: pd.Series(dict(n=len(d), speech_acc=100 * (d.speech_pred == d.true).mean(),
                                     fusion_acc=100 * (d.fusion_pred == d.true).mean(),
                                     gate=d.gate_speech.mean() if "gate_speech" in d else np.nan)),
            include_groups=False)
        g["delta"] = g.fusion_acc - g.speech_acc
        g.round(3).to_csv(RESULTS_DIR / f"fusion_gain_by_speech_confidence_{var}_si.csv")
        lines += [f"## SI: gain vs. speech-only confidence ({var})", "", g.round(2).to_markdown(), ""]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        x = np.arange(len(g))
        ax.bar(x - 0.2, g.speech_acc, 0.4, label="speech-only", color="#0072B2")
        ax.bar(x + 0.2, g.fusion_acc, 0.4, label=f"fusion ({var.split('_')[-1].upper()})", color="#E69F00")
        ax.set_xticks(x, [f"{iv.left:.2f}-{min(iv.right, 1):.2f}\n(n={int(n)})" for iv, n in zip(g.index, g.n)],
                      fontsize=7)
        ax.set_xlabel("speech-only model confidence (max softmax)")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"Speaker-independent: where fusion helps ({var})", fontsize=9)
        ax.legend(fontsize=8)
        fig.savefig(PLOTS_DIR / f"fusion_gain_vs_confidence_{var}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3) gate distribution
    fig, ax = plt.subplots(figsize=(6, 3))
    for var, c in (("gated_wavlm_gt", "#0072B2"), ("gated_wavlm_asr", "#D55E00")):
        gs = [pd.read_csv(f).gate_speech for f in (RUNS_DIR / "fusion" / var).glob("*/test_predictions.csv")]
        if gs:
            ax.hist(pd.concat(gs), bins=40, range=(0, 1), alpha=0.6, color=c,
                    label=f"{var.split('_')[-1].upper()} transcripts", density=True)
    ax.set_xlabel("mean GMU gate z per clip (1 = speech, 0 = text)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title("How much the fused representation draws on speech vs. text", fontsize=9)
    fig.savefig(PLOTS_DIR / "fusion_gate_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    (RESULTS_DIR / "fusion_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# ------------------------------------------------------------------ learned layer weights
def layer_weight_plot():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, (pipe, var, key, xl) in zip(axes, [
            ("speech", "wavlm", "encoder.frontend", [f"L{i}" for i in (0, 2, 4, 6, 8, 10, 12)]),
            ("text", "gt", "encoder.frontend", [f"L{i}" for i in range(7)])]):
        W = [json.loads(f.read_text()).get("layer_weights", {}).get(key)
             for f in (RUNS_DIR / pipe / var).glob("*/test_metrics.json")]
        W = np.array([w for w in W if w])
        if not len(W):
            continue
        ax.bar(range(W.shape[1]), W.mean(0), yerr=W.std(0), color="#0072B2", capsize=3)
        ax.set_xticks(range(W.shape[1]), xl)
        ax.set_title(f"{pipe} encoder layer weights ({len(W)} runs)", fontsize=9)
        ax.set_ylabel("softmax weight")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "learned_layer_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ ASR transcript statistics
def asr_stats(df_idx):
    f = cached("asr_transcripts.csv")
    if not f.exists():
        return
    from common.data import clean_transcript
    a = pd.read_csv(f).fillna("")
    a = a.merge(df_idx[["uid", "speaker", "emotion", "word"]], on="uid")
    a["clean"] = a.asr_raw.map(clean_transcript)
    a["ends_q"] = a.clean.str.endswith("?")
    a["ends_excl"] = a.clean.str.endswith("!")
    a["ends_period"] = a.clean.str.endswith(".")
    a["word_correct"] = [w in c.replace("?", " ").replace("!", " ").replace(".", " ").replace(",", " ").split()
                         for w, c in zip(a.word, a.clean)]
    a["exact_carrier"] = a.clean.str.replace(r"[.,?!]", "", regex=True).str.strip() == ("say the word " + a.word)
    t = a.groupby("emotion")[["ends_q", "ends_excl", "ends_period", "word_correct", "exact_carrier"]].mean() * 100
    t.round(1).to_csv(RESULTS_DIR / "asr_transcript_stats_by_emotion.csv")
    a[["uid", "speaker", "emotion", "asr_raw"]].to_csv(RESULTS_DIR / "asr_transcripts.csv", index=False)
    md = ["# Whisper-small transcripts of TESS: what survives into text", "",
          "% of clips per emotion whose transcript ends in ?, !, ., contains the correct target word, "
          "or is exactly the carrier sentence.", "", t.round(1).to_markdown(), "",
          "Examples:", ""]
    for e in EMOTIONS:
        ex = a[a.emotion == e].asr_raw.head(4).tolist()
        md.append(f"- **{EMOTION_NAMES[e]}**: " + " | ".join(f"`{x}`" for x in ex))
    (RESULTS_DIR / "asr_transcript_stats.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


# ------------------------------------------------------------------ statistics
def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def calibration():
    rows = []
    for pipe, var, name in MAIN:
        for proto in PROTOS:
            vals = []
            for f in sorted((RUNS_DIR / pipe / var).glob(f"{proto}_*/test_predictions.csv")):
                d = pd.read_csv(f)
                P = d[[f"p_{e}" for e in EMOTIONS]].values
                y = d.true.map(EMOTIONS.index).values
                nll = float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())
                vals.append((ece(P, y), nll, float(P.max(1).mean())))
            if vals:
                v = np.array(vals)
                rows.append(dict(model=name, protocol=proto, ECE=100 * v[:, 0].mean(), NLL=v[:, 1].mean(),
                                 mean_confidence=100 * v[:, 2].mean(), runs=len(v)))
    t = pd.DataFrame(rows)
    t.round(3).to_csv(RESULTS_DIR / "calibration.csv", index=False)
    return t


def significance():
    """Paired comparisons on identical splits/seeds: per-run accuracy differences
    (mean, 95% t-CI, paired t-test, Wilcoxon) and a pooled exact McNemar test on clips."""
    from scipy import stats
    pairs = [(("speech", "logmel"), ("speech", "wavlm")),
             (("speech", "wavlm"), ("fusion", "gated_wavlm_gt")),
             (("speech", "wavlm"), ("fusion", "gated_wavlm_asr")),
             (("speech", "wavlm"), ("fusion", "concat_wavlm_gt")),
             (("speech", "wavlm"), ("fusion", "xattn_wavlm_gt")),
             (("speech", "wavlm"), ("speech", "wavlm_spknorm")),
             (("fusion", "gated_wavlm_asr"), ("fusion", "gated_wavlm_asr_spknorm")),
             (("fusion", "gated_wavlm_gt"), ("fusion", "gated_wavlm_asr"))]
    rows = []
    for (pa, va), (pb, vb) in pairs:
        for proto in PROTOS:
            diffs, b01, b10 = [], 0, 0
            for f in sorted((RUNS_DIR / pa / va).glob(f"{proto}_*/test_predictions.csv")):
                g = RUNS_DIR / pb / vb / f.parent.name / "test_predictions.csv"
                if not g.exists():
                    continue
                A, B = pd.read_csv(f), pd.read_csv(g)
                ca, cb = (A.pred == A.true).values, (B.pred == B.true).values
                diffs.append(cb.mean() - ca.mean())
                b01 += int((~ca & cb).sum())
                b10 += int((ca & ~cb).sum())
            if not diffs:
                continue
            d = np.array(diffs) * 100
            n = len(d)
            ci = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            p_t = stats.ttest_1samp(d, 0).pvalue if n > 1 and d.std() > 0 else np.nan
            p_w = stats.wilcoxon(d).pvalue if n >= 5 and np.any(d != 0) else np.nan
            p_mc = stats.binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 > 0 else 1.0
            rows.append(dict(A=f"{pa}/{va}", B=f"{pb}/{vb}", protocol=proto, runs=n,
                             delta_acc_pp=d.mean(), ci95_pp=ci, p_paired_t=p_t, p_wilcoxon=p_w,
                             B_fixes=b01, B_breaks=b10, p_mcnemar_pooled=p_mc))
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS_DIR / "significance.csv", index=False)
    md = ["# Paired significance tests (B vs A on identical splits and seeds)", "",
          "delta = mean per-run accuracy difference B - A (percentage points) with 95 % t-CI over runs; "
          "McNemar pooled over all test clips of all runs (B_fixes = A wrong & B right).", "",
          t.round(4).to_markdown(index=False), ""]
    cal = calibration()
    md += ["# Calibration (test set, mean over runs)", "", "ECE (15 bins, %), NLL, mean max-probability (%).", "",
           cal.round(3).to_markdown(index=False), ""]
    (RESULTS_DIR / "statistics.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df_idx = load_index()
    runs = collect()
    if runs.empty:
        raise SystemExit("no evaluated runs found")
    accuracy_tables(runs, df_idx)
    confusion_plots(runs)
    layer_weight_plot()
    asr_stats(df_idx)
    fusion_analysis(runs)
    significance()
    print(separability(df_idx).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
