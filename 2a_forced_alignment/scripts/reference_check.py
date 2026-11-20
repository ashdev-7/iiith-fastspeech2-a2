"""An aligner-independent reference for pauses, computed from the signal alone.

There is no hand-labelled reference for this corpus, so agreement between aligners measures consistency,
not accuracy. Pauses are the one thing that can be located without any aligner and without a human:
stretches where Praat's intensity contour stays at the file's noise-floor level.

For every pause of >= MIN_PAUSE s:
  * pause time in words: how much of that stretch the aligner assigns to a word (robust metric);
  * edge error: signed distance from the pause start (speech offset) to the aligner's nearest word end,
    and from the pause end (speech onset) to the nearest word start (threshold-sensitive, see below).
Everything is repeated at three thresholds. Result: the pause-absorption metric is stable across
thresholds, but the edge errors are not. An energy detector also calls breaths, clicks and noise bursts
"sound", and the threshold decides where a fading word ends, so it is not precise enough to rank
aligners at the 10-20 ms level.

    python scripts/reference_check.py            (needs praat-parselmouth)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_alignments import read_textgrid  # noqa: E402

EXPERIMENTS = ["arpa_base", "arpa_adapted", "arpa_g2p_custom", "mfa_ipa", "mfa_ipa_dict_v300"]
THRESHOLDS_DB = (6.0, 9.0, 12.0)       # pause = intensity within this many dB of the file's noise floor
MIN_PAUSE = 0.15                        # s; longer than almost all stop closures
MIN_SOUNDING = 0.05
OUT = ROOT / "analysis"


def praat_silences(wav: Path, above_floor_db: float) -> tuple[list[tuple[float, float]], float]:
    """Pauses = stretches where Praat's intensity contour stays within `above_floor_db` of the file's
    noise floor (10th percentile of intensity) for >= MIN_PAUSE s; sounding gaps < MIN_SOUNDING s are
    bridged. A pause is a stretch at background-noise level, so the threshold is relative to the floor."""
    snd = parselmouth.Sound(str(wav))
    it = snd.to_intensity(minimum_pitch=100.0, time_step=0.005)
    t, v = it.xs(), it.values[0]
    silent = v < np.percentile(v, 10) + above_floor_db
    runs, i = [], 0                                   # run-length encode
    while i < len(t):
        j = i
        while j + 1 < len(t) and silent[j + 1] == silent[i]:
            j += 1
        runs.append([silent[i], t[i], t[j]])
        i = j + 1
    for k in range(1, len(runs) - 1):                 # bridge short sounding blips inside pauses
        if not runs[k][0] and runs[k][2] - runs[k][1] < MIN_SOUNDING and runs[k - 1][0] and runs[k + 1][0]:
            runs[k][0] = True
    merged = []
    for r in runs:
        if merged and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
        else:
            merged.append(list(r))
    dt = 0.0025                                       # half a frame: interval edges between frame centres
    out = [(max(0.0, a - dt), min(snd.duration, b + dt)) for s_, a, b in merged if s_ and b - a + 2 * dt >= MIN_PAUSE]
    if out and out[0][0] <= 0.03:
        out[0] = (0.0, out[0][1])
    if out and out[-1][1] >= snd.duration - 0.03:
        out[-1] = (out[-1][0], snd.duration)
    return out, snd.duration


def words(tg: Path) -> list[tuple[float, float]]:
    return [(a, b) for a, b, t in read_textgrid(tg)["words"] if t and t not in ("<eps>", "sil", "sp")]


def main():
    wavs = sorted((ROOT / "corpus").glob("*/*.wav"))
    rows, pauses = [], []
    for thr in THRESHOLDS_DB:
        for wav in wavs:
            sil, dur = praat_silences(wav, thr)
            for exp in EXPERIMENTS:
                tg = ROOT / "outputs" / exp / wav.parent.name / f"{wav.stem}.TextGrid"
                if not tg.exists():
                    continue
                w = words(tg)
                ends = np.array([b for _, b in w])
                starts = np.array([a for a, _ in w])
                for a, b in sil:
                    covered = sum(max(0.0, min(b, wb) - max(a, wa)) for wa, wb in w)
                    pauses.append(dict(threshold_db=thr, file=wav.stem, experiment=exp, start=a, end=b,
                                       pause_s=b - a, in_word_frac=covered / (b - a)))
                    if a > 0.01:   # speech before the pause: offset landmark
                        e = ends[np.argmin(np.abs(ends - a))] - a
                        rows.append(dict(threshold_db=thr, file=wav.stem, experiment=exp, kind="offset",
                                         landmark=a, error_ms=1000 * e))
                    if b < dur - 0.01:  # speech after the pause: onset landmark
                        e = starts[np.argmin(np.abs(starts - b))] - b
                        rows.append(dict(threshold_db=thr, file=wav.stem, experiment=exp, kind="onset",
                                         landmark=b, error_ms=1000 * e))
    e, p = pd.DataFrame(rows), pd.DataFrame(pauses)
    e.to_csv(OUT / "reference_pause_edges.csv", index=False)
    p.to_csv(OUT / "reference_pauses.csv", index=False)

    e["abs"] = e.error_ms.abs()
    e["speaker"] = np.where(e.file.str.startswith("ISLE"), "isle", "f2b")
    p["speaker"] = np.where(p.file.str.startswith("ISLE"), "isle", "f2b")
    summ = e.groupby(["threshold_db", "speaker", "experiment"], sort=False).agg(
        edges=("abs", "size"), median_abs_ms=("abs", "median"), mean_abs_ms=("abs", "mean"),
        within_20ms=("abs", lambda x: 100 * (x <= 20.0001).mean()),
        within_50ms=("abs", lambda x: 100 * (x <= 50.0001).mean()), max_abs_ms=("abs", "max")).round(1)
    rec = p.groupby(["threshold_db", "speaker", "experiment"], sort=False).agg(
        pauses=("in_word_frac", "size"), pauses_kept_pct=("in_word_frac", lambda x: 100 * (x < 0.5).mean()),
        pause_time_in_words_pct=("in_word_frac", "mean")).round(3)
    rec["pause_time_in_words_pct"] = (100 * rec.pause_time_in_words_pct).round(1)
    table = summ.join(rec)

    # precision of the reference itself: how far does each landmark move between thresholds?
    base = e[e.experiment == "arpa_base"]
    mid = THRESHOLDS_DB[1]
    drift = []
    for r in base[base.threshold_db == mid].itertuples():
        pos = [r.landmark]
        for thr in THRESHOLDS_DB:
            if thr == mid:
                continue
            c = base[(base.threshold_db == thr) & (base.file == r.file) & (base.kind == r.kind)].landmark.values
            if len(c) and np.min(np.abs(c - r.landmark)) < 0.2:
                pos.append(c[np.argmin(np.abs(c - r.landmark))])
        if len(pos) == len(THRESHOLDS_DB):
            drift.append(1000 * (max(pos) - min(pos)))
    drift = np.array(drift)

    txt = ["# Aligner-independent reference: pauses found from the Praat intensity contour", "",
           "Pauses are found from Praat's intensity contour (`Sound: To Intensity`, 100 Hz, 5 ms step): stretches of "
           f">= {MIN_PAUSE * 1000:.0f} ms whose intensity stays within "
           f"{', '.join(f'{t:g}' for t in THRESHOLDS_DB)} dB of the file's noise floor (10th percentile), "
           f"bridging sounding blips < {MIN_SOUNDING * 1000:.0f} ms. No aligner or human is involved.", "",
           "* `edges`: speech offsets/onsets at the detected pauses; error = aligner's nearest word end/start "
           "minus the Praat landmark.",
           "* `pauses_kept_pct`: share of detected pauses that the aligner leaves (mostly) outside words; "
           "`pause_time_in_words_pct`: mean share of pause time the aligner assigns to words.", "",
           f"**Precision of the reference:** a landmark found at several thresholds moves by a median of "
           f"{np.median(drift):.0f} ms (90th percentile {np.percentile(drift, 90):.0f} ms, n = {len(drift)}) "
           "between them, because the energy threshold decides where a fading word ends. "
           "The edge errors also change a lot with the threshold, since an energy detector calls breaths, clicks "
           "and noise bursts \"sound\". **Use this reference for pause absorption only** "
           "(`pause_time_in_words_pct`, stable across thresholds), not to rank aligners at the 10-20 ms level.", "",
           table.to_markdown(), ""]
    (OUT / "reference_check.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
