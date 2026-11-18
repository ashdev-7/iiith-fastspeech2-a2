"""Organise the provided dataset into the layout MFA expects.

MFA corpus format (one sub-directory per speaker, a .lab transcript next to each .wav):

    corpus/
      f2b/            F2BJRLP1.wav  F2BJRLP1.lab  ...    (BU Radio News, speaker F2B)
      isle_sess0131/  ISLE_..._01_sprt1.wav  ISLE_..._01_sprt1.lab ...   (ISLE L2 learner)

Text normalisation applied (every change is written to corpus/normalization_log.tsv):
  * line breaks / repeated spaces -> single spaces; CRLF stripped
  * "seventy- six" (number split across a line/space)   -> "seventy six"
  * acronyms read letter-by-letter are spelled out:   S.J.C.'s -> s j c's,  WBUR's -> w b u r's
  * "de-politicize" -> "depoliticize" (a single lexical word; the hyphen is orthographic)
  * punctuation other than word-internal apostrophes is removed, text lower-cased
Audio: all files are 16 kHz / mono / 16-bit PCM, but the three BU Radio News
files are NIST SPHERE despite their .wav extension; they are rewritten as RIFF
WAV (sample data untouched). ISLE files are already RIFF and are copied as-is.

    python scripts/prepare_corpus.py --raw data/raw --out corpus
"""
from __future__ import annotations

import argparse
import re
import shutil
import wave
from pathlib import Path

SPEAKER_OF = {"F2B": "f2b", "ISLE_SESS0131": "isle_sess0131"}

# explicit, auditable rewrite rules (applied in order, case-insensitive)
RULES = [
    (r"\bS\.J\.C\.'s", "s j c's"),
    (r"\bWBUR's", "w b u r's"),
    (r"\bde-politicize\b", "depoliticize"),
    (r"(\w+)-\s+(\w+)", r"\1 \2"),     # "seventy- six" -> "seventy six"
]


def normalise(text: str) -> tuple[str, list[tuple[str, str]]]:
    changes = []
    text = text.replace("\r", " ").replace("\n", " ")
    for pat, rep in RULES:
        for m in re.finditer(pat, text, flags=re.I):
            changes.append((m.group(0), re.sub(pat, rep, m.group(0), flags=re.I)))
        text = re.sub(pat, rep, text, flags=re.I)
    text = re.sub(r"[^\w' ]+", " ", text)           # drop . , : etc. (keep apostrophes)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)    # stray quotes
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text, changes


def read_nist_sphere(path: Path) -> tuple[bytes, int, int, int]:
    """Minimal NIST SPHERE (NIST_1A) reader for uncompressed PCM.

    The BU Radio News files ship with a `.wav` extension but are SPHERE files
    (1024-byte ASCII header, raw little-endian PCM). Several tools mis-read
    them, so we rewrite them as canonical RIFF/WAVE. Returns (pcm_bytes, sr, channels, sampwidth).
    """
    raw = path.read_bytes()
    assert raw[:7] == b"NIST_1A", path
    hdr_size = int(raw[8:16].split()[0])
    fields = {}
    for line in raw[16:hdr_size].decode("ascii", "replace").splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3:
            fields[parts[0]] = parts[2]
        if line.strip() == "end_head":
            break
    assert fields.get("sample_coding", "pcm").startswith("pcm"), f"compressed SPHERE not supported: {path}"
    width, sr, ch = int(fields["sample_n_bytes"]), int(fields["sample_rate"]), int(fields["channel_count"])
    n = int(fields["sample_count"])
    pcm = raw[hdr_size: hdr_size + n * width * ch]
    if fields.get("sample_byte_format") == "10" and width == 2:   # big-endian -> little-endian
        pcm = b"".join(pcm[i + 1:i + 2] + pcm[i:i + 1] for i in range(0, len(pcm), 2))
    return pcm, sr, ch, width


def copy_as_riff_wav(src: Path, dst: Path) -> tuple[str, float]:
    with open(src, "rb") as f:
        magic = f.read(4)
    if magic == b"RIFF":
        shutil.copy2(src, dst)
        fmt = "RIFF/WAVE"
    else:
        pcm, sr, ch, width = read_nist_sphere(src)
        with wave.open(str(dst), "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(width)
            w.setframerate(sr)
            w.writeframes(pcm)
        fmt = "NIST SPHERE -> RIFF/WAVE"
    with wave.open(str(dst)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), dst
        return fmt, w.getnframes() / w.getframerate()


def speaker_for(stem: str) -> str:
    for prefix, spk in SPEAKER_OF.items():
        if stem.upper().startswith(prefix):
            return spk
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="corpus")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    log = ["file\toriginal\tnormalised"]
    for wav in sorted((raw / "wav").glob("*.wav")):
        trs = [p for p in (raw / "transcripts").iterdir() if p.stem.lower() == wav.stem.lower()]
        assert len(trs) == 1, f"no unique transcript for {wav.name}"
        text, changes = normalise(trs[0].read_text(encoding="utf-8", errors="replace"))
        spk_dir = out / speaker_for(wav.stem)
        spk_dir.mkdir(parents=True, exist_ok=True)
        fmt, dur = copy_as_riff_wav(wav, spk_dir / wav.name)
        (spk_dir / f"{wav.stem}.lab").write_text(text + "\n", encoding="utf-8")
        log += [f"{wav.stem}\t{a}\t{b}" for a, b in changes]
        print(f"{spk_dir.name:14s} {wav.stem:34s} {dur:6.2f}s {len(text.split()):3d} words [{fmt}]")
    (out / "normalization_log.tsv").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
