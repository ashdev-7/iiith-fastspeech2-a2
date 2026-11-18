"""Sanity checks on data preparation and on MFA's own repeatability.

1. Audio identity   - every prepared corpus .wav carries exactly the provided samples
                      (NIST SPHERE payload vs. RIFF data, or byte-identical copy).
2. Untouched input  - the arpa_base setup run on the provided files *as given*
                      (outputs/arpa_untouched) vs. the submitted arpa_base alignment:
                      which words differ and what MFA does with them.
3. Noise floor      - arpa_base re-run on the same prepared corpus (outputs/repeatability/run*):
                      how much MFA's boundaries move between identical runs. Every other
                      comparison is reported next to it.

Outputs (analysis/): sanity_audio_identity.csv, sanity_untouched_diffs.csv,
sanity_boundary_agreement.csv, sanity_checks.md

    python scripts/sanity_checks.py
"""
from __future__ import annotations

import difflib
import hashlib
import re
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_alignments import ANA, CORPUS, OUT, ROOT, SIL, read_textgrid  # noqa: E402

RAW = ROOT / "data" / "raw" / "wav"


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    cols = list(df.columns)
    fmt = lambda v: f"{v:.1f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# ------------------------------------------------------------------ 1. audio
def read_riff(path: Path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return x, w.getframerate(), w.getnchannels(), w.getsampwidth()


def audio_identity() -> pd.DataFrame:
    rows = []
    for wav in sorted(CORPUS.rglob("*.wav")):
        raw = (RAW / wav.name).read_bytes()
        x, sr, ch, width = read_riff(wav)
        row = dict(file=wav.stem, provided_format="RIFF/WAVE", converted_rate=sr, converted_channels=ch,
                   converted_bytes_per_sample=width, converted_samples=len(x))
        if raw[:7] == b"NIST_1A":
            hlen = int(raw[8:16].decode().strip())
            hdr = raw[:hlen].decode("ascii", "ignore")
            field = lambda k: re.search(rf"{k} -\w+ (\S+)", hdr).group(1)  # noqa: E731
            assert field("sample_coding") == "pcm" and field("sample_byte_format") == "01" and field("sample_n_bytes") == "2"
            orig = np.frombuffer(raw[hlen:], dtype="<i2")
            row.update(provided_format="NIST SPHERE (pcm, 16-bit LE)", header_rate=int(field("sample_rate")),
                       header_samples=int(field("sample_count")), header_max=int(field("sample_max")),
                       header_min=int(field("sample_min")), converted_max=int(x.max()), converted_min=int(x.min()),
                       samples_identical=bool(np.array_equal(orig, x)))
        else:
            row.update(samples_identical=hashlib.md5(raw).digest() == hashlib.md5(wav.read_bytes()).digest())
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ alignments
def load(exp_dir: Path, keep_spn: bool = False) -> dict[str, dict]:
    drop = SIL - {"spn"} if keep_spn else SIL
    out = {}
    for tg in sorted(exp_dir.rglob("*.TextGrid")):
        t = read_textgrid(tg)
        out[tg.stem] = {tier: [iv for iv in ivs if iv[2].lower() not in drop] for tier, ivs in t.items()}
        out[tg.stem]["speaker"] = tg.parent.name
    return out


def matched_diffs(a: list, b: list):
    """Boundary differences (ms) over label-matched intervals, plus fraction of `a` matched."""
    sm = difflib.SequenceMatcher(None, [x[2] for x in a], [y[2] for y in b], autojunk=False)
    d, n = [], 0
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            x, y = a[blk.a + k], b[blk.b + k]
            d += [abs(x[0] - y[0]), abs(x[1] - y[1])]
            n += 1
    return np.array(d) * 1000, n / max(len(a), 1)


def agreement_table(pairs: dict[str, tuple], exclude: dict[str, dict] | None = None) -> pd.DataFrame:
    """exclude: {comparison: {file: [(t0, t1), ...]}} - drop intervals overlapping these spans
    (used to compare only the words whose text is identical in both inputs)."""
    rows = []
    for name, (pa, pb) in pairs.items():
        A, B = load(pa), load(pb)
        for f, spans in (exclude or {}).get(name, {}).items():
            keep = lambda iv: all(iv[1] <= a + 1e-6 or iv[0] >= b - 1e-6 for a, b in spans)  # noqa: E731
            for X in (A, B):
                if f in X:
                    X[f] = {k: ([iv for iv in v if keep(iv)] if k != "speaker" else v) for k, v in X[f].items()}
        if not A or not B:
            continue
        for tier in ("words", "phones"):
            for group in ("f2b", "isle"):
                d, cov = [], []
                for f in A:
                    if f in B and A[f]["speaker"].startswith(group):
                        di, c = matched_diffs(A[f][tier], B[f][tier])
                        d.append(di)
                        cov.append(c)
                if not d:
                    continue
                d = np.concatenate(d)
                if not len(d):  # e.g. IPA vs ARPA phone labels never match
                    continue
                rows.append(dict(comparison=name, tier=tier, speaker=group, boundaries=len(d),
                                 matched_pct=100 * np.mean(cov), identical_pct=100 * np.mean(d < 0.5),
                                 within_10ms_pct=100 * np.mean(d <= 10.5), within_20ms_pct=100 * np.mean(d <= 20.5),
                                 mean_ms=d.mean(), max_ms=d.max()))
    return pd.DataFrame(rows)


def untouched_diffs() -> pd.DataFrame:
    A, B = load(OUT / "arpa_base", keep_spn=True), load(OUT / "arpa_untouched", keep_spn=True)
    rows = []
    for f in A:
        wa, wb = A[f]["words"], B[f]["words"]
        sm = difflib.SequenceMatcher(None, [w[2] for w in wa], [w[2] for w in wb], autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue
            t0 = min([w[0] for w in wa[i1:i2] + wb[j1:j2]])
            t1 = max([w[1] for w in wa[i1:i2] + wb[j1:j2]])
            def ph(P, W):  # phones inside this side's own word span
                if not W:
                    return ""
                s0, s1 = W[0][0], W[-1][1]
                return " ".join(p[2] for p in P if s0 - 1e-6 <= p[0] < s1 - 1e-6)
            rows.append(dict(file=f, start=round(t0, 3), end=round(t1, 3),
                             prepared_words=" ".join(w[2] for w in wa[i1:i2]),
                             untouched_words=" ".join(w[2] for w in wb[j1:j2]),
                             prepared_phones=ph(A[f]["phones"], wa[i1:i2]),
                             untouched_phones=ph(B[f]["phones"], wb[j1:j2])))
    return pd.DataFrame(rows)


def main():
    md = ["# Sanity checks", ""]
    aud = audio_identity()
    aud.to_csv(ANA / "sanity_audio_identity.csv", index=False)
    md += ["## 1. Audio identity (prepared corpus vs. provided files)", "",
           f"All files sample-identical: **{bool(aud.samples_identical.all())}**", "",
           md_table(aud), ""]

    ud = untouched_diffs()
    ud.to_csv(ANA / "sanity_untouched_diffs.csv", index=False)
    md += ["## 2. Words that differ when the provided files are aligned untouched", "",
           "`spn` = MFA's spoken-noise phone, used for words not in the dictionary (no phone-level alignment).", "",
           md_table(ud) if len(ud) else "(none)", ""]

    reps = sorted((OUT / "repeatability").glob("run*"))
    pairs = {f"repeat: arpa_base vs {r.name}": (OUT / "arpa_base", r) for r in reps}
    if len(reps) >= 2:
        pairs[f"repeat: {reps[0].name} vs {reps[1].name}"] = (reps[0], reps[1])
    pairs.update({"untouched input vs arpa_base": (OUT / "arpa_base", OUT / "arpa_untouched"),
                  "arpa_adapted vs arpa_base": (OUT / "arpa_base", OUT / "arpa_adapted"),
                  "arpa_g2p_custom vs arpa_base": (OUT / "arpa_base", OUT / "arpa_g2p_custom")})
    excl = {}
    for r in ud.itertuples():
        excl.setdefault(r.file, []).append((r.start, r.end))
    ag = agreement_table(pairs, {"untouched input vs arpa_base": excl})
    ag.round(2).to_csv(ANA / "sanity_boundary_agreement.csv", index=False)
    md += ["## 3. Boundary agreement, with MFA's run-to-run noise floor", "",
           "Boundaries of label-matched intervals (silences excluded; for the untouched input, the spans of the "
           "words listed in section 2 are excluded, so only words with identical text are compared). `matched_pct` = share of reference "
           "intervals whose label sequence is matched. 10 ms = one MFA frame.", "",
           md_table(ag), ""]
    (ANA / "sanity_checks.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
