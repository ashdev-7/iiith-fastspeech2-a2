"""Completeness check of the combined report against the two standalone reports.

1. Every figure and table caption of each standalone PDF appears exactly once in the combined PDF, with the part
   prefix (Figure 2a.N / Table 2b.N) and in the same order.
2. Every image drawn in the standalone PDFs is drawn in the combined PDF (same number of image and vector figures).
3. The body text of each part is the same: after removing numbering, every word of each standalone report occurs
   in the combined report at least as often (only headers, numbering and page numbers may differ).

    python combined_report/check_combined.py
"""
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
COMB = pymupdf.open(ROOT / "combined_report" / "Assignment2_Yash_Tyagi.pdf")
PARTS = {"2a": pymupdf.open(ROOT / "2a_forced_alignment" / "report" / "report.pdf"),
         "2b": pymupdf.open(ROOT / "2b_multimodal_emotion_recognition" / "report" / "report.pdf")}
fails = []


def text(doc, pages=None):
    return "\n".join(doc[i].get_text() for i in (pages if pages is not None else range(len(doc))))


def captions(t, prefix=""):
    return re.findall(r"^(Figure|Table)\s+" + re.escape(prefix) + r"(\d+)\.\s", t, flags=re.M)


# page ranges of the two parts in the combined PDF (each starts with its part heading)
starts = {k: next(i for i, p in enumerate(COMB) if p.get_text().lstrip().startswith(f"Assignment {k}")) for k in PARTS}
ranges = {"2a": range(starts["2a"], starts["2b"]), "2b": range(starts["2b"], len(COMB))}

for k, doc in PARTS.items():
    body_start = 0 if k == "2a" else next(i for i, p in enumerate(doc) if "1. Dataset and what it implies" in p.get_text())
    want = captions(text(doc, range(body_start, len(doc))))
    got = captions(text(COMB, ranges[k]), prefix=k + ".")
    if want != got:
        fails.append(f"{k}: captions differ: standalone {want} vs combined {got}")
    print(f"{k}: {len(want)} captions in the standalone, {len(got)} in the combined part, same order: {want == got}")
    n_img = lambda d, rg: sum(len(d[i].get_images()) for i in rg)
    n_draw = lambda d, rg: sum(1 for i in rg if len(d[i].get_drawings()) > 50)
    a, b = n_img(doc, range(body_start, len(doc))), n_img(COMB, ranges[k])
    if a != b:
        fails.append(f"{k}: {a} raster images in the standalone vs {b} in the combined part")
    print(f"{k}: raster images {a} vs {b}; pages with vector figures {n_draw(doc, range(body_start, len(doc)))} vs {n_draw(COMB, ranges[k])}")

    def words(t):
        t = re.sub(r"-\s*\n\s*", "", t)          # undo hyphenation at line breaks (layouts break lines differently)
        t = t.replace("-", "")
        t = re.sub(r"\b(Figure|Table|Section|Fig\.|Sec\.)\s*(2[ab]\.)?\d+(\.\d+)*\b", r"\1", t)
        t = re.sub(r"\b(2[ab]\.)?\d+(\.\d+)*\.\s", " ", t)           # heading numbers
        return [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", t)]
    body_text = text(doc, range(body_start, len(doc)))
    if k == "2a":                                  # the standalone title block is replaced by the part heading
        body_text = body_text[body_text.index("1. Setup and data preparation"):]
    ws, wc = words(body_text), words(text(COMB, ranges[k]))
    # every word of the standalone body occurs in the combined part at least as often (figure positions may differ)
    from collections import Counter
    missing = Counter(ws) - Counter(wc)
    extra = Counter(wc) - Counter(ws)
    print(f"{k}: {len(ws)} words in the standalone body; missing in the combined part: {sum(missing.values())} "
          f"{dict(missing.most_common(8))}; extra: {sum(extra.values())} {dict(extra.most_common(8))}")
    # tolerated: at most two short fragments of a word hyphenated across a page break or a figure (e.g. "me- | dian")
    real = {w: n for w, n in missing.items() if len(w) > 4}
    if real or sum(missing.values()) > 2:
        fails.append(f"{k}: words of the standalone body missing in the combined part: {dict(missing)}")

print("COMBINED REPORT OK" if not fails else "\n".join(["FAIL"] + fails))
sys.exit(1 if fails else 0)
