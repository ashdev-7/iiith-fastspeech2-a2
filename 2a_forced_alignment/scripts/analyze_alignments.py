"""Inspect and compare MFA alignments (no gold reference is available, so all
checks are reference-free or cross-model).

Outputs (analysis/):
  intervals_<exp>.csv            every word / phone interval of every file
  summary_by_experiment.csv      counts, durations, silence ratio, suspicious-phone counts
  agreement_vs_arpa_base.csv     word (and phone) boundary agreement between experiments
  pronunciation_diffs.csv        words whose chosen pronunciation differs between dictionaries
  energy_flags.csv               words aligned to near-silence / silences that contain speech
  figures/*.png                  Praat-style waveform + spectrogram + tier plots, model overlays

    python scripts/analyze_alignments.py
"""
from __future__ import annotations

import re
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS, OUT, ANA = ROOT / "corpus", ROOT / "outputs", ROOT / "analysis"
FIG = ANA / "figures"
SIL = {"", "sil", "sp", "spn", "<eps>", "silence"}
EXP_ORDER = ["arpa_base", "mfa_ipa", "arpa_g2p_custom", "arpa_adapted"]
EXP_LABEL = {"arpa_base": "english_us_arpa", "mfa_ipa": "english_mfa (IPA)",
             "arpa_g2p_custom": "arpa + G2P-only dict", "arpa_adapted": "arpa adapted"}
SHORT_PHONE = 0.0301  # <= 3 frames (30 ms): very short, typically reduced/compressed phones
# (MFA 3 models allow 1-frame = 10 ms phones, so these are the candidates for "skipped" phones)


# ------------------------------------------------------------------ IO
def read_textgrid(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    """Parse a long- or short-format Praat TextGrid -> {tier: [(xmin, xmax, text)]}."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    tiers = {}
    if "item [" in txt:  # long format
        for block in re.split(r"\n\s*item \[\d+\]:", txt)[1:]:
            name = re.search(r'name = "(.*?)"', block).group(1)
            ivs = re.findall(r'xmin = ([\d.eE+-]+)\s*\n\s*xmax = ([\d.eE+-]+)\s*\n\s*text = "(.*?)"', block, re.S)
            tiers[name] = [(float(a), float(b), t.strip()) for a, b, t in ivs]
    else:  # short format
        toks = re.findall(r'"(?:[^"]|"")*"|[-\d.eE+]+', txt)
        i = toks.index('"IntervalTier"') if '"IntervalTier"' in toks else None
        while i is not None and i < len(toks):
            name = toks[i + 1].strip('"')
            n = int(toks[i + 4])
            ivs, j = [], i + 5
            for _ in range(n):
                ivs.append((float(toks[j]), float(toks[j + 1]), toks[j + 2].strip('"')))
                j += 3
            tiers[name] = ivs
            i = j if j < len(toks) and toks[j] == '"IntervalTier"' else None
    # normalise tier names ("f2b - words" -> "words")
    return {("words" if "word" in k.lower() else "phones" if "phone" in k.lower() else k): v
            for k, v in tiers.items()}


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return x, sr


def load_experiments() -> dict[str, pd.DataFrame]:
    exps = {}
    for exp in EXP_ORDER:
        rows = []
        for tg in sorted((OUT / exp).rglob("*.TextGrid")):
            for tier, ivs in read_textgrid(tg).items():
                for s, e, lab in ivs:
                    rows.append(dict(file=tg.stem, speaker=tg.parent.name, tier=tier, start=s, end=e,
                                     label=lab, dur=e - s, is_sil=lab.lower() in SIL))
        if rows:
            exps[exp] = pd.DataFrame(rows)
            exps[exp].to_csv(ANA / f"intervals_{exp}.csv", index=False)
    return exps


# ------------------------------------------------------------------ analyses
def summary(exps):
    rows = []
    for exp, df in exps.items():
        for spk, d in [("all", df)] + list(df.groupby("speaker")):
            w, p = d[d.tier == "words"], d[d.tier == "phones"]
            ws, ps = w[~w.is_sil], p[~p.is_sil]
            total = d[d.tier == "words"].groupby("file").end.max().sum()
            rows.append(dict(
                experiment=exp, speaker=spk, files=d.file.nunique(), words=len(ws), phones=len(ps),
                unk_spn=int((w.label == "<unk>").sum() + (p.label == "spn").sum()),
                silence_pct=100 * w[w.is_sil].dur.sum() / total,
                mean_phone_ms=1000 * ps.dur.mean(), median_phone_ms=1000 * ps.dur.median(),
                short_phones_le30ms=int((ps.dur <= SHORT_PHONE).sum()),
                short_phones_le30ms_pct=100 * (ps.dur <= SHORT_PHONE).mean(),
                long_phones_250ms=int((ps.dur > 0.25).sum()),
                mean_word_ms=1000 * ws.dur.mean()))
    out = pd.DataFrame(rows)
    out.round(2).to_csv(ANA / "summary_by_experiment.csv", index=False)
    return out


def mfa_quality():
    """Merge MFA's own per-utterance alignment_analysis.csv across experiments.
    Log-likelihoods are only comparable between models sharing a phone set/topology
    (arpa_base vs arpa_g2p_custom vs arpa_adapted), not with the IPA model."""
    frames = []
    for exp in EXP_ORDER:
        f = OUT / exp / "alignment_analysis.csv"
        if f.exists():
            frames.append(pd.read_csv(f).assign(experiment=exp))
    if not frames:
        return pd.DataFrame()
    q = pd.concat(frames)
    wide = q.pivot_table(index="file", columns="experiment",
                         values=["speech_log_likelihood", "phone_duration_deviation"])
    wide.columns = [f"{a}|{b}" for a, b in wide.columns]
    wide.round(3).to_csv(ANA / "mfa_quality_by_experiment.csv")
    return q


def boundaries(df, tier):
    """Per file: list of (label, start, end) for non-silence intervals."""
    d = df[(df.tier == tier) & ~df.is_sil]
    return {f: list(zip(g.label, g.start, g.end)) for f, g in d.groupby("file")}


def agreement(exps, ref="arpa_base"):
    rows = []
    for exp, df in exps.items():
        if exp == ref:
            continue
        tiers = ["words"] + (["phones"] if "arpa" in exp else [])  # phone sets differ for IPA
        for tier in tiers:
            A, B = boundaries(exps[ref], tier), boundaries(df, tier)
            for f in sorted(set(A) & set(B)):
                a, b = A[f], B[f]
                if len(a) != len(b):
                    rows.append(dict(experiment=exp, tier=tier, file=f, n=len(a), n_other=len(b),
                                     mean_abs_ms=np.nan, median_abs_ms=np.nan, within_20ms_pct=np.nan,
                                     within_50ms_pct=np.nan, max_abs_ms=np.nan))
                    continue
                diffs = np.abs(np.array([[x[1] - y[1], x[2] - y[2]] for x, y in zip(a, b)])).ravel() * 1000
                rows.append(dict(experiment=exp, tier=tier, file=f, n=len(a), n_other=len(b),
                                 mean_abs_ms=diffs.mean(), median_abs_ms=np.median(diffs),
                                 within_20ms_pct=100 * (diffs <= 20).mean(),
                                 within_50ms_pct=100 * (diffs <= 50).mean(), max_abs_ms=diffs.max()))
    out = pd.DataFrame(rows)
    out.round(2).to_csv(ANA / "agreement_vs_arpa_base.csv", index=False)
    return out


def word_prons(df):
    """(file, word index) -> phone string chosen by the aligner."""
    res = {}
    for f, g in df.groupby("file"):
        words = g[(g.tier == "words") & ~g.is_sil]
        phones = g[(g.tier == "phones") & ~g.is_sil]
        for i, (lab, s, e) in enumerate(zip(words.label, words.start, words.end)):
            ph = phones[(phones.start >= s - 1e-4) & (phones.end <= e + 1e-4)].label
            res[(f, i, lab)] = " ".join(ph)
    return res


def pron_diffs(exps):
    if not {"arpa_base", "arpa_g2p_custom"} <= set(exps):
        return pd.DataFrame()
    A, B = word_prons(exps["arpa_base"]), word_prons(exps["arpa_g2p_custom"])
    rows = [dict(file=k[0], word=k[2], lexicon_pron=A[k], g2p_pron=B.get(k, "?"))
            for k in A if B.get(k) != A[k]]
    out = pd.DataFrame(rows).drop_duplicates(["word", "lexicon_pron", "g2p_pron"])
    out.to_csv(ANA / "pronunciation_diffs.csv", index=False)
    return out


def frame_energy_db(f):
    """10 ms frame energy in dB relative to the file's 95th-percentile frame."""
    x, sr = read_wav(next(CORPUS.rglob(f"{f}.wav")))
    hop = int(0.01 * sr)
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (0, 400)), 400)[::hop]
    e_db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    e_db = e_db - np.percentile(e_db, 95)
    floor = np.percentile(e_db, 10)          # per-file noise floor (recordings differ in SNR)
    return e_db, floor


def energy_flags(exps):
    """Reference-free sanity check on every word / silence interval of every experiment.
      * word longer than 0.25 s with > 40 % of its frames within 6 dB of the file's
        noise floor -> word stretched over a pause (boundary error)
      * word whose mean energy is < -35 dB -> word placed on silence
      * silence > 0.15 s with mean energy > -15 dB -> silence that swallowed speech"""
    rows = []
    cache = {}
    for exp, df in exps.items():
        for f, g in df[df.tier == "words"].groupby("file"):
            e_db, floor = cache.setdefault(f, frame_energy_db(f))
            for lab, s, e, sil in zip(g.label, g.start, g.end, g.is_sil):
                seg = e_db[int(round(s * 100)): max(int(round(s * 100)) + 1, int(round(e * 100)))]
                mean_db = 10 * np.log10(np.mean(10 ** (seg / 10)) + 1e-12)
                quiet = float((seg < floor + 6).mean())
                flag = ""
                if not sil and mean_db < -35:
                    flag = "word on near-silence"
                elif not sil and quiet > 0.4 and e - s > 0.25:
                    flag = "word stretched over a pause"
                elif sil and e - s > 0.15 and mean_db > -15:
                    flag = "silence containing speech energy"
                rows.append(dict(experiment=exp, file=f, label=lab or "<sil>", start=s, end=e, dur=e - s,
                                 rel_energy_db=mean_db, quiet_frac=quiet, flag=flag))
    out = pd.DataFrame(rows)
    out.round(3).to_csv(ANA / "energy_all_intervals.csv", index=False)
    out[out.flag != ""].round(3).to_csv(ANA / "energy_flags.csv", index=False)
    return out


# ------------------------------------------------------------------ figures
def spectrogram(ax, x, sr, t0, t1):
    seg = x[int(t0 * sr): int(t1 * sr)]
    # 16 ms Hann window / 2.5 ms hop ("broadband" like Praat's default), 50 dB dynamic range
    from matplotlib import mlab
    S, _, _ = mlab.specgram(seg, NFFT=256, Fs=sr, noverlap=256 - 40, window=np.hanning(256))
    S_db = 10 * np.log10(S + 1e-12)
    top = np.percentile(S_db, 99.5)
    ax.imshow(S_db, origin="lower", aspect="auto", cmap="Greys", vmin=top - 50, vmax=top,
              extent=(t0, t1, 0, sr / 2))
    ax.set_ylim(0, 5000)
    ax.set_ylabel("Hz")


def draw_tier(ax, ivs, t0, t1, name, fontsize=8, color="#1f4e79"):
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    for s, e, lab in ivs:
        if e < t0 or s > t1:
            continue
        ax.axvline(s, color=color, lw=0.8)
        ax.axvline(e, color=color, lw=0.8)
        if lab and lab.lower() not in SIL:
            ax.text((max(s, t0) + min(e, t1)) / 2, 0.5, lab, ha="center", va="center", fontsize=fontsize,
                    clip_on=True)


def plot_alignment(exp, df, f, t0=None, t1=None, tag=""):
    wav = next(CORPUS.rglob(f"{f}.wav"))
    x, sr = read_wav(wav)
    t0 = 0 if t0 is None else t0
    t1 = len(x) / sr if t1 is None else t1
    g = df[df.file == f]
    tiers = {t: list(zip(d.start, d.end, d.label)) for t, d in g.groupby("tier")}
    fig, axes = plt.subplots(4, 1, figsize=(13, 6.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.1, 2, 0.6, 0.6], hspace=0.05))
    tt = np.arange(int(t0 * sr), int(t1 * sr)) / sr
    axes[0].plot(tt, x[int(t0 * sr): int(t1 * sr)], lw=0.4, color="k")
    axes[0].set_yticks([])
    spectrogram(axes[1], x, sr, t0, t1)
    for s, e, _ in tiers.get("words", []):
        if t0 <= s <= t1:
            axes[1].axvline(s, color="#c00000", lw=0.6, alpha=0.6)
    draw_tier(axes[2], tiers.get("words", []), t0, t1, "words", 9)
    draw_tier(axes[3], tiers.get("phones", []), t0, t1, "phones", 7, "#7f6000")
    axes[3].set_xlabel("time (s)")
    axes[0].set_title(f"{f}  |  {EXP_LABEL.get(exp, exp)}  |  {t0:.2f}-{t1:.2f} s", fontsize=10)
    fig.savefig(FIG / f"align_{exp}_{f}{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(exps, f, t0=None, t1=None, tag=""):
    """Word boundaries of every experiment over one spectrogram."""
    wav = next(CORPUS.rglob(f"{f}.wav"))
    x, sr = read_wav(wav)
    t0 = 0 if t0 is None else t0
    t1 = len(x) / sr if t1 is None else t1
    names = [e for e in EXP_ORDER if e in exps]
    fig, axes = plt.subplots(1 + len(names), 1, figsize=(13, 2.2 + 0.6 * len(names)), sharex=True,
                             gridspec_kw=dict(height_ratios=[2] + [0.6] * len(names), hspace=0.05))
    spectrogram(axes[0], x, sr, t0, t1)
    colors = ["#1f4e79", "#c00000", "#548235", "#7030a0"]
    for ax, exp, c in zip(axes[1:], names, colors):
        d = exps[exp]
        ivs = list(zip(*[d[(d.file == f) & (d.tier == "words")][k] for k in ("start", "end", "label")]))
        draw_tier(ax, ivs, t0, t1, EXP_LABEL[exp], 8, c)
        for s, e, lab in ivs:
            if lab and lab.lower() not in SIL and t0 <= s <= t1:
                axes[0].axvline(s, color=c, lw=0.7, alpha=0.7)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(f"{f}: word boundaries from all experiments ({t0:.2f}-{t1:.2f} s)", fontsize=10)
    fig.savefig(FIG / f"overlay_{f}{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_phone_durations(exps):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bins = np.arange(0, 0.31, 0.01)
    for exp in [e for e in EXP_ORDER if e in exps]:
        d = exps[exp]
        p = d[(d.tier == "phones") & ~d.is_sil].dur.clip(upper=0.3)
        ax.hist(p, bins=bins, histtype="step", lw=1.5, label=EXP_LABEL[exp])
    ax.axvline(SHORT_PHONE, color="k", ls=":", lw=1)
    ax.text(SHORT_PHONE + 0.003, ax.get_ylim()[1] * 0.9, "30 ms", fontsize=8)
    ax.set_xlabel("phone duration (s, clipped at 0.3)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.set_title("Phone duration distributions")
    fig.savefig(FIG / "phone_duration_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    exps = load_experiments()
    if not exps:
        raise SystemExit("no TextGrids under outputs/ - run the align stage first")
    pd.set_option("display.width", 200)
    print("\n== summary ==\n", summary(exps).round(2).to_string(index=False))
    ag = agreement(exps)
    if not ag.empty:
        print("\n== boundary agreement vs arpa_base (per file) ==\n", ag.round(1).to_string(index=False))
        print("\n== boundary agreement (all files pooled, mean of file means) ==\n",
              ag.groupby(["experiment", "tier"])[["mean_abs_ms", "median_abs_ms", "within_20ms_pct"]]
              .mean().round(1).to_string())
    q = mfa_quality()
    if not q.empty:
        print("\n== MFA alignment_analysis: mean per experiment ==\n",
              q.groupby(["experiment", "speaker"])[["speech_log_likelihood", "phone_duration_deviation", "snr"]]
              .mean().round(3).to_string())
    pdif = pron_diffs(exps)
    if not pdif.empty:
        print("\n== pronunciation differences lexicon vs G2P-only ==\n", pdif.to_string(index=False))
    en = energy_flags(exps)
    print("\n== energy flags (all experiments) ==\n", en[en.flag != ""].round(3).to_string(index=False))
    print("\n== flag counts ==\n", en[en.flag != ""].groupby(["experiment", "flag"]).size().to_string())
    plot_phone_durations(exps)
    for exp, df in exps.items():
        for f in sorted(df.file.unique()):
            if f.startswith("F2B"):
                plot_alignment(exp, df, f, 0, 6, "_0-6s")
            else:
                plot_alignment(exp, df, f)
    files = sorted(next(iter(exps.values())).file.unique())
    for f in files:
        if f.startswith("F2B"):
            dur = next(iter(exps.values())).query("file == @f").end.max()
            for a in range(0, int(np.ceil(dur)), 6):
                plot_overlay(exps, f, a, min(a + 6, dur), f"_{a:02d}s")
        else:
            plot_overlay(exps, f)
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
