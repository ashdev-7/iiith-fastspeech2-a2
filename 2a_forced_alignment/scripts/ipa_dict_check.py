"""Is english_mfa's poor alignment of the ISLE learner speech caused by its dictionary?

Compares mfa_ipa (english_mfa acoustic model + v3.1.0 single-pronunciation dictionary, the MFA
default) with mfa_ipa_dict_v300 (same acoustic model + full v3.0.0 dictionary with variants), both
against arpa_base. Outputs analysis/ipa_dict_check.md and analysis/ipa_dict_check_words.csv.

    python scripts/ipa_dict_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_alignments import ANA, OUT, SIL, read_textgrid  # noqa: E402
from sanity_checks import agreement_table, md_table  # noqa: E402

EXPS = {"mfa_ipa": "english_mfa + v3.1.0 dict (MFA default)",
        "mfa_ipa_dict_v300": "english_mfa + v3.0.0 dict (all variants)",
        "arpa_base": "english_us_arpa (reference)"}


def word_rows() -> pd.DataFrame:
    rows = []
    for exp in EXPS:
        for tg in sorted((OUT / exp).rglob("*.TextGrid")):
            t = read_textgrid(tg)
            total = t["words"][-1][1]
            sil = sum(e - s for s, e, lab in t["words"] if lab.lower() in SIL)
            for s, e, lab in t["words"]:
                if lab.lower() in SIL or not tg.stem.startswith("ISLE"):
                    continue
                ph = " ".join(p[2] for p in t["phones"] if s - 1e-6 <= p[0] < e - 1e-6)
                rows.append(dict(experiment=exp, utterance=tg.stem.split("_")[3], word=lab, start=round(s, 2),
                                 end=round(e, 2), dur_s=round(e - s, 2), phones=ph,
                                 file_silence_pct=round(100 * sil / total, 1)))
    return pd.DataFrame(rows)


def main():
    w = word_rows()
    w.to_csv(ANA / "ipa_dict_check_words.csv", index=False)
    piv = w.pivot_table(index=["utterance", "word", "start"], columns="experiment", values="dur_s", aggfunc="first")
    last = w.groupby(["experiment", "utterance"]).tail(1)[["experiment", "utterance", "word", "start", "end", "dur_s",
                                                          "phones", "file_silence_pct"]]
    prons = w[w.word.isin(["said", "white", "not", "bait", "bed", "bad", "new", "no"])] \
        .drop_duplicates(["experiment", "word"])[["experiment", "word", "phones"]]
    ag = agreement_table({f"{e} vs arpa_base": (OUT / "arpa_base", OUT / e) for e in ("mfa_ipa", "mfa_ipa_dict_v300")})
    ag = ag[ag.tier == "words"]
    md = ["# english_mfa: v3.1.0 (default) vs. v3.0.0 dictionary", "",
          "\n".join(f"* `{k}`: {v}" for k, v in EXPS.items()), "",
          "## Pronunciations chosen on the ISLE prompts", "", md_table(prons), "",
          "## Last word of each ISLE prompt (where mfa_ipa absorbed the trailing silence)", "", md_table(last), "",
          "## Word-boundary agreement with arpa_base", "", md_table(ag), ""]
    (ANA / "ipa_dict_check.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
