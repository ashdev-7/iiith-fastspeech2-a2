"""Invariant tests for the forced-alignment outputs (silent errors that would not crash anything).

  * every experiment aligned every file, and its word tier spells exactly the transcript
  * tiers are gap-free and cover the whole recording; boundaries sit on MFA's 10 ms grid
  * every word interval is exactly the span of its phones
  * no out-of-vocabulary (spn) words in the submitted experiments; every word is in the dictionary
  * the converted audio is sample-identical to the provided files
  * the published summary table recomputes from the TextGrids

    python -m pytest tests/ -q        (or: python tests/test_invariants.py)
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_alignments import read_textgrid  # noqa: E402

EXPERIMENTS = ["arpa_base", "arpa_adapted", "arpa_g2p_custom", "mfa_ipa", "mfa_ipa_dict_v300"]
WAVS = sorted((ROOT / "corpus").glob("*/*.wav"))
SIL = {"", "sil", "sp", "<eps>"}
DICTS = {"arpa_base": "english_us_arpa_plus_g2p.dict", "arpa_adapted": "english_us_arpa_plus_g2p.dict",
         "arpa_g2p_custom": "g2p_custom_arpa.dict", "mfa_ipa": "english_mfa_plus_g2p.dict",
         "mfa_ipa_dict_v300": "english_mfa_v3.0.0_plus_g2p.dict"}


def dur(wav: Path) -> float:
    with wave.open(str(wav)) as w:
        return w.getnframes() / w.getframerate()


def tg(exp: str, wav: Path):
    return read_textgrid(ROOT / "outputs" / exp / wav.parent.name / f"{wav.stem}.TextGrid")


def test_six_files_per_experiment():
    assert len(WAVS) == 6
    for exp in EXPERIMENTS:
        for w in WAVS:
            assert (ROOT / "outputs" / exp / w.parent.name / f"{w.stem}.TextGrid").exists(), (exp, w.stem)


def test_word_tier_spells_the_transcript():
    for exp in EXPERIMENTS:
        for w in WAVS:
            words = [t for _, _, t in tg(exp, w)["words"] if t not in SIL]
            lab = w.with_suffix(".lab").read_text(encoding="utf-8").split()
            assert words == lab, (exp, w.stem, [(a, b) for a, b in zip(words, lab) if a != b][:3])


def test_tiers_are_contiguous_cover_the_file_and_sit_on_the_grid():
    for exp in EXPERIMENTS:
        for w in WAVS:
            d = dur(w)
            for tier, ivs in tg(exp, w).items():
                starts = np.array([a for a, _, _ in ivs])
                ends = np.array([b for _, b, _ in ivs])
                assert abs(starts[0]) < 1e-6 and abs(ends[-1] - d) < 0.011, (exp, w.stem, tier, ends[-1], d)
                assert np.allclose(starts[1:], ends[:-1]), (exp, w.stem, tier, "gap or overlap")
                assert (ends > starts).all(), (exp, w.stem, tier, "empty interval")
                inner = ends[:-1]  # the last boundary is the file end, not a frame boundary
                assert np.allclose(inner * 100, np.round(inner * 100), atol=1e-6), (exp, w.stem, tier, "off 10 ms grid")


def test_word_intervals_are_spans_of_their_phones():
    for exp in EXPERIMENTS:
        for w in WAVS:
            g = tg(exp, w)
            pb = {round(a, 4) for a, _, _ in g["phones"]} | {round(b, 4) for _, b, _ in g["phones"]}
            for a, b, t in g["words"]:
                assert round(a, 4) in pb and round(b, 4) in pb, (exp, w.stem, t)
                inside = [p for pa, pbb, p in g["phones"] if pa >= a - 1e-6 and pbb <= b + 1e-6]
                if t not in SIL:
                    assert inside and all(p not in SIL for p in inside), (exp, w.stem, t, inside)


def test_no_oov_and_every_word_in_the_dictionary():
    for exp in EXPERIMENTS:
        lex = set()
        for line in (ROOT / "dictionaries" / DICTS[exp]).read_text(encoding="utf-8").splitlines():
            if line.strip():
                lex.add(line.split()[0].lower())
        for w in WAVS:
            g = tg(exp, w)
            assert not any(p == "spn" for _, _, p in g["phones"]), (exp, w.stem, "spoken-noise phone")
            for _, _, t in g["words"]:
                if t in SIL:
                    continue
                base = t[:-2] if t.endswith("'s") and t not in lex else t
                assert t in lex or base in lex, (exp, w.stem, t)


def test_converted_audio_is_sample_identical():
    t = pd.read_csv(ROOT / "analysis" / "sanity_audio_identity.csv")
    assert len(t) == 6 and t.samples_identical.all(), t[["file", "samples_identical"]]


def test_summary_table_recomputes():
    pub = pd.read_csv(ROOT / "analysis" / "summary_by_experiment.csv").set_index(["experiment", "speaker"])
    for exp in ("arpa_base", "mfa_ipa", "arpa_g2p_custom", "arpa_adapted"):
        for spk in ("f2b", "isle_sess0131"):
            ws = [w for w in WAVS if w.parent.name == spk]
            phones = [(a, b, p) for w in ws for a, b, p in tg(exp, w)["phones"]]
            speech = [(a, b) for a, b, p in phones if p not in SIL]
            total = sum(dur(w) for w in ws)
            sil = total - sum(b - a for a, b in speech)
            key = (exp, "f2b" if spk == "f2b" else "isle")
            row = pub.loc[key] if key in pub.index else pub.loc[(exp, spk)]
            assert row.phones == len(speech), (exp, spk, row.phones, len(speech))
            assert abs(row.silence_pct - 100 * sil / total) < 0.6, (exp, spk, row.silence_pct, 100 * sil / total)
            assert abs(row.mean_phone_ms - 1000 * np.mean([b - a for a, b in speech])) < 0.06, (exp, spk)


def test_report_delta_column_recomputes():
    """The report's 'word boundary difference vs arpa_base' column, pooled over all word boundaries."""
    import re
    lines = (ROOT / "report" / "report.tex").read_text(encoding="utf-8").splitlines()
    for exp in ("mfa_ipa", "arpa_g2p_custom", "arpa_adapted"):
        label = exp.replace("_", r"\_")
        i = [k for k, ln in enumerate(lines) if ln.startswith(label + " ") and "& f2b" in ln][0]
        assert "& isle" in lines[i + 1], (exp, lines[i + 1])
        for spk, line in (("f2b", lines[i]), ("isle_sess0131", lines[i + 1])):
            shown = float(re.findall(r"([\d.]+)\\,ms", line)[0])
            d = []
            for w in [w for w in WAVS if w.parent.name == spk]:
                a = [(x, y) for x, y, t in tg("arpa_base", w)["words"] if t not in SIL]
                b = [(x, y) for x, y, t in tg(exp, w)["words"] if t not in SIL]
                d += [abs(p - q) for (p, _), (q, _) in zip(a, b)] + [abs(p - q) for (_, p), (_, q) in zip(a, b)]
            tol = 0.05 if shown != round(shown) else 0.5  # "167 ms" is printed to the nearest ms
            assert abs(shown - 1000 * np.mean(d)) <= tol + 1e-6, (exp, spk, shown, 1000 * np.mean(d))


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
