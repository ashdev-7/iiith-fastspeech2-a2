"""Inspect the alignments with Praat itself, headless, via parselmouth (Praat 6.1 embedded).

1. Renders Praat figures (waveform + spectrogram + TextGrid tiers) with draw_alignment.praat,
   exactly as `Praat --run` would: analysis/figures/praat_*.png
2. Measures, with Praat's own Intensity and Pitch analyses, the stretches where english_mfa
   labels a word but english_us_arpa labels silence (Error 1 in the report), compared with the
   speech inside the same file: analysis/praat_measurements.csv

    pip install praat-parselmouth
    python scripts/praat/run_praat.py

Interactive inspection (listening) still needs the Praat GUI: open the .wav and the .TextGrid,
select both, View & Edit, select an interval and press Tab to play it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import parselmouth
from parselmouth.praat import call

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_alignments import SIL, read_textgrid  # noqa: E402

FIG = ROOT / "analysis" / "figures"
SCRIPT = ROOT / "scripts" / "praat" / "draw_alignment.praat"
ISLE = "isle_sess0131/ISLE_SESS0131_BLOCKD02_0{}_sprt1"

FIGURES = [  # (experiment, utterance, t_start, t_end, tag)
    ("arpa_base", ISLE.format(1), 0, 2.5, ""),  # speech ends at ~2.3 s
    ("arpa_base", ISLE.format(3), 0, 0, ""),
    ("mfa_ipa", ISLE.format(3), 0, 0, ""),
    ("arpa_base", "f2b/F2BJRLP2", 6, 9, "_6-9s"),
    ("mfa_ipa", "f2b/F2BJRLP2", 6, 9, "_6-9s"),
    ("arpa_base", "f2b/F2BJRLP1", 20, 22, "_20-22s"),
]


def draw():
    """Headless Praat can only export vector graphics, so draw to EPS and convert with epstopdf
    (TeX Live / Ghostscript). The figure itself is drawn entirely by Praat."""
    import shutil
    import subprocess
    for exp, utt, t0, t1, tag in FIGURES:
        eps = FIG / f"praat_{exp}_{Path(utt).name}{tag}.eps"
        parselmouth.praat.run_file(str(SCRIPT), str(ROOT / "corpus" / f"{utt}.wav"),
                                   str(ROOT / "outputs" / exp / f"{utt}.TextGrid"), str(eps), float(t0), float(t1))
        if shutil.which("epstopdf"):
            subprocess.check_call(["epstopdf", eps.name], cwd=FIG)
            eps.unlink()
            print("wrote", eps.with_suffix(".pdf").relative_to(ROOT))
        else:
            print("wrote", eps.relative_to(ROOT), "(install TeX Live/Ghostscript for PDF)")


def words(exp, utt):
    return [w for w in read_textgrid(ROOT / "outputs" / exp / f"{utt}.TextGrid")["words"] if w[2].lower() not in SIL]


def measure():
    """Praat intensity / voicing of the extra stretch each english_mfa word covers beyond arpa_base."""
    rows = []
    for utt in [ISLE.format(i) for i in (1, 2, 3)] + ["f2b/F2BJRLP2"]:
        snd = parselmouth.Sound(str(ROOT / "corpus" / f"{utt}.wav"))
        inten = snd.to_intensity(time_step=0.01)
        pitch = snd.to_pitch(time_step=0.01)
        base = words("arpa_base", utt)
        speech_db = [call(inten, "Get mean", s, e, "energy") for s, e, _ in base]
        ref = sum(speech_db) / len(speech_db)
        for s, e, lab in words("mfa_ipa", utt):
            match = [b for b in base if b[2] == lab and b[0] < e and b[1] > s]
            if not match:
                continue
            b = match[0]
            for a, z in ((s, b[0]), (b[1], e)):  # parts of the IPA word outside the ARPA word
                if z - a < 0.1:
                    continue
                db = call(inten, "Get mean", a, z, "energy")
                voiced = call(pitch, "Count voiced frames") and sum(
                    1 for k in range(1, call(pitch, "Get number of frames") + 1)
                    if a <= call(pitch, "Get time from frame number", k) < z
                    and call(pitch, "Get value in frame", k, "Hertz") == call(pitch, "Get value in frame", k, "Hertz"))
                n = max(1, round((z - a) / 0.01))
                rows.append(dict(utterance=Path(utt).name, word=lab, extra_start=round(a, 2), extra_end=round(z, 2),
                                 extra_dur_s=round(z - a, 2), extra_mean_dB=round(db, 1),
                                 speech_words_mean_dB=round(ref, 1), dB_below_speech=round(ref - db, 1),
                                 extra_voiced_pct=round(100 * voiced / n, 1)))
    import csv
    out = ROOT / "analysis" / "praat_measurements.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    print("Praat", parselmouth.PRAAT_VERSION)
    draw()
    measure()
