"""End-to-end, re-runnable MFA forced-alignment pipeline.

Stages (each can be run alone with --stages):
  prepare   organise raw data into the MFA corpus layout            (scripts/prepare_corpus.py)
  models    download pretrained acoustic models, dictionaries, G2P models
  oov       find out-of-vocabulary words per dictionary
  dict      build the dictionaries used in the experiments:
              <base>_plus_g2p.dict : pretrained dictionary + G2P pronunciations for OOVs
              g2p_custom_arpa.dict : a dictionary generated *entirely* from the transcripts by G2P
  validate  `mfa validate` the corpus against the final dictionary
  align     run every experiment in EXPERIMENTS -> outputs/<exp>/**/*.TextGrid
  adapt     speaker-adapt english_us_arpa on this corpus (MAP) and align with it
  analyze   tables + figures                                         (scripts/analyze_alignments.py)
  sanity    audio-identity check, alignment of the untouched provided files, MFA run-to-run
            repeatability, and every comparison judged against that noise floor (scripts/sanity_checks.py)
  ipa_dict  english_mfa acoustic model with the full v3.0.0 english_mfa dictionary instead of the
            single-pronunciation v3.1.0 default: are mfa_ipa's errors caused by the lexicon? (scripts/ipa_dict_check.py)

Run inside the conda env that has MFA installed:
    python scripts/run_pipeline.py                 # everything
    python scripts/run_pipeline.py --stages align analyze
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
DICTS = ROOT / "dictionaries"
OUT = ROOT / "outputs"
LOGS = ROOT / "logs"
MODELS_DIR = ROOT / "models"
MFA_ROOT = Path(os.environ.get("MFA_ROOT_DIR", Path.home() / "Documents" / "MFA"))
PRETRAINED = MFA_ROOT / "pretrained_models"

# name -> (dictionary, acoustic model, description)
EXPERIMENTS = {
    "arpa_base":       ("dictionaries/english_us_arpa_plus_g2p.dict", "english_us_arpa",
                        "english_us_arpa acoustic model + english_us_arpa dictionary (+G2P for OOVs)"),
    "mfa_ipa":         ("dictionaries/english_mfa_plus_g2p.dict", "english_mfa",
                        "english_mfa (IPA) acoustic model + english_mfa dictionary (+G2P for OOVs)"),
    "arpa_g2p_custom": ("dictionaries/g2p_custom_arpa.dict", "english_us_arpa",
                        "english_us_arpa acoustic model + custom dictionary generated from transcripts by G2P"),
}
ADAPTED = ("arpa_adapted", "dictionaries/english_us_arpa_plus_g2p.dict", "english_us_arpa",
           "english_us_arpa acoustic model MAP-adapted on this corpus")

# beam: the BU Radio utterances are 25-31 s long; a wider beam avoids alignment failures
ALIGN_OPTS = ["--clean", "--overwrite", "--beam", "100", "--retry_beam", "400", "-j", "4"]


def mfa(*args, log_name: str):
    LOGS.mkdir(exist_ok=True)
    cmd = ["mfa", *map(str, args)]
    print("$", " ".join(cmd), flush=True)
    t0 = time.time()
    with open(LOGS / f"{log_name}.log", "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    print(f"  -> exit {rc} in {time.time() - t0:.1f}s (log: logs/{log_name}.log)", flush=True)
    if rc != 0:
        print((LOGS / f"{log_name}.log").read_text(encoding="utf-8")[-3000:])
        raise SystemExit(rc)


def corpus_words() -> list[str]:
    words = set()
    for lab in CORPUS.rglob("*.lab"):
        words.update(lab.read_text(encoding="utf-8").split())
    return sorted(words)


def read_dict_words(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {line.split()[0].lower() for line in f if line.strip()}


def pretrained_dict(name: str) -> Path:
    return PRETRAINED / "dictionary" / f"{name}.dict"


# ------------------------------------------------------------------ stages
def stage_prepare():
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "prepare_corpus.py"),
                           "--raw", str(ROOT / "data" / "raw"), "--out", str(CORPUS)])


REQUIRED_MODELS = [("acoustic", "english_us_arpa", "zip"), ("acoustic", "english_mfa", "zip"),
                   ("dictionary", "english_us_arpa", "dict"), ("dictionary", "english_mfa", "dict"),
                   ("g2p", "english_us_arpa", "zip"), ("g2p", "english_us_mfa", "zip")]
G2P_FOR_DICT = {"english_us_arpa": "english_us_arpa", "english_mfa": "english_us_mfa"}


def stage_models():
    # `mfa model download` queries the GitHub API (60 requests/h unauthenticated);
    # skip models that are already present so re-runs never hit the limit.
    for kind, name, ext in REQUIRED_MODELS:
        if (PRETRAINED / kind / f"{name}.{ext}").exists():
            print(f"  {kind}/{name} already present")
            continue
        mfa("model", "download", kind, name, log_name=f"download_{kind}_{name}")
    mfa("version", log_name="mfa_version")


def in_vocab(word: str, vocab: set[str]) -> bool:
    """MFA splits English clitics ('s, 'll, ...) off before lookup, so "state's" is
    covered when "state" and "'s" are; mirror that so G2P is only used for true OOVs."""
    if word in vocab:
        return True
    m = re.fullmatch(r"(.+)('s|'ll|'re|'ve|'d|'m|n't)", word)
    return bool(m) and m.group(1) in vocab and m.group(2) in vocab


def stage_oov():
    DICTS.mkdir(exist_ok=True)
    words = corpus_words()
    for name in ("english_us_arpa", "english_mfa"):
        vocab = read_dict_words(pretrained_dict(name))
        oov = [w for w in words if not in_vocab(w, vocab)]
        (DICTS / f"oov_{name}.txt").write_text("\n".join(oov) + "\n", encoding="utf-8")
        print(f"{name}: {len(words)} corpus word types, {len(oov)} OOV -> {oov}")


def stage_dict():
    for name in ("english_us_arpa", "english_mfa"):
        oov_file = DICTS / f"oov_{name}.txt"
        g2p_out = DICTS / f"g2p_oov_{name}.dict"
        if oov_file.read_text().strip():
            mfa("g2p", oov_file, G2P_FOR_DICT[name], g2p_out, "--num_pronunciations", "1", "--clean",
                log_name=f"g2p_oov_{name}")
        else:
            g2p_out.write_text("")
        merged = DICTS / f"{name}_plus_g2p.dict"
        shutil.copyfile(pretrained_dict(name), merged)
        with open(merged, "a", encoding="utf-8") as f:
            f.write("\n" + g2p_out.read_text(encoding="utf-8"))
    # custom dictionary trained/generated from the transcripts alone (no lexicon lookup)
    words = DICTS / "corpus_wordlist.txt"
    words.write_text("\n".join(corpus_words()) + "\n", encoding="utf-8")
    mfa("g2p", words, "english_us_arpa", DICTS / "g2p_custom_arpa.dict", "--num_pronunciations", "2",
        "--clean", log_name="g2p_custom_arpa")
    # small, readable excerpt of the dictionary entries actually used by the corpus
    used = set(corpus_words())
    for d in ("english_us_arpa_plus_g2p", "english_mfa_plus_g2p"):
        lines = [l for l in (DICTS / f"{d}.dict").read_text(encoding="utf-8").splitlines()
                 if l.strip() and l.split()[0].lower() in used]
        (DICTS / f"{d}.corpus_subset.dict").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_validate():
    mfa("validate", CORPUS, DICTS / "english_us_arpa_plus_g2p.dict", "--acoustic_model_path", "english_us_arpa",
        "--clean", "-j", "4", log_name="validate_arpa")


def stage_align():
    for exp, (dct, am, desc) in EXPERIMENTS.items():
        print(f"\n### {exp}: {desc}")
        mfa("align", CORPUS, ROOT / dct, am, OUT / exp, *ALIGN_OPTS, log_name=f"align_{exp}")


def stage_adapt():
    exp, dct, am, desc = ADAPTED
    print(f"\n### {exp}: {desc}")
    MODELS_DIR.mkdir(exist_ok=True)
    adapted = MODELS_DIR / "english_us_arpa_adapted.zip"
    mfa("adapt", CORPUS, ROOT / dct, am, adapted, *ALIGN_OPTS, log_name=f"adapt_{exp}")
    # NB: `mfa adapt --output_directory` exports the alignments computed *before*
    # adaptation ("Alignment already done, skipping"), so align again with the adapted model.
    mfa("align", CORPUS, ROOT / dct, adapted, OUT / exp, *ALIGN_OPTS, log_name=f"align_{exp}")


def stage_analyze():
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "analyze_alignments.py")])
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "report_figures.py")])  # report figures, print size


CORPUS_UNTOUCHED = ROOT / "corpus_untouched"
REPEATS = 2


def build_untouched_corpus():
    """The provided files exactly as given (NIST SPHERE audio, original transcript text),
    only placed in MFA's speaker folders with the transcript extension renamed to .lab."""
    raw = ROOT / "data" / "raw"
    if CORPUS_UNTOUCHED.exists():
        shutil.rmtree(CORPUS_UNTOUCHED)
    for wav in sorted(CORPUS.rglob("*.wav")):
        spk = CORPUS_UNTOUCHED / wav.parent.name
        spk.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw / "wav" / wav.name, spk / wav.name)
        txt = next(p for p in (raw / "transcripts").iterdir() if p.stem == wav.stem)
        shutil.copyfile(txt, spk / f"{wav.stem}.lab")


def stage_sanity():
    """Does data preparation change the alignment, and how repeatable is MFA itself?
      outputs/arpa_untouched/          arpa_base setup on the untouched provided files
      outputs/repeatability/run<k>/    arpa_base setup re-run on the prepared corpus
    The submitted experiments are not re-run (MFA is not bit-for-bit repeatable)."""
    build_untouched_corpus()
    dct, am, _ = EXPERIMENTS["arpa_base"]
    mfa("align", CORPUS_UNTOUCHED, ROOT / dct, am, OUT / "arpa_untouched", *ALIGN_OPTS,
        log_name="align_arpa_untouched")
    for k in range(1, REPEATS + 1):
        mfa("align", CORPUS, ROOT / dct, am, OUT / "repeatability" / f"run{k}", *ALIGN_OPTS,
            log_name=f"align_repeatability_run{k}")
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "sanity_checks.py")])


IPA_DICT_V300 = ("dictionary-english_mfa-v3.0.0/english_mfa.dict", "english_mfa_v3.0.0",
                 "2b99a13da9e92bb4bcb7f0dcefb05ff4")


def stage_ipa_dict():
    """english_mfa acoustic model with the *previous* english_mfa dictionary release (v3.0.0).
    The current release (v3.1.0, the MFA default used in mfa_ipa) keeps a single pronunciation per
    word, some of them rare variants (e.g. said -> s aj d, white -> ʋ aj ʈ); v3.0.0 has all variants
    with probabilities (said -> s ɛ d, p = 0.99). Tests whether mfa_ipa's errors come from the lexicon."""
    import hashlib
    import urllib.request
    rel, name, md5 = IPA_DICT_V300
    path = PRETRAINED / "dictionary" / f"{name}.dict"
    if not path.exists():
        url = "https://github.com/MontrealCorpusTools/mfa-models/releases/download/" + rel
        print("downloading", url)
        path.write_bytes(urllib.request.urlopen(url, timeout=300).read())
    assert hashlib.md5(path.read_bytes()).hexdigest() == md5, f"{path} does not match the official release"
    vocab = read_dict_words(path)
    oov = [w for w in corpus_words() if not in_vocab(w, vocab)]
    print(f"{name}: OOV -> {oov}")
    g2p = {l.split("\t")[0]: l for l in (DICTS / "g2p_oov_english_mfa.dict").read_text(encoding="utf-8").splitlines()
           if l.strip()}
    missing = [w for w in oov if w not in g2p]
    if missing:  # G2P exactly as in stage_dict, only for OOVs not already generated
        (DICTS / f"oov_{name}.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
        mfa("g2p", DICTS / f"oov_{name}.txt", G2P_FOR_DICT["english_mfa"], DICTS / f"g2p_oov_{name}.dict",
            "--num_pronunciations", "1", "--clean", log_name=f"g2p_oov_{name}")
        g2p.update({l.split("\t")[0]: l for l in (DICTS / f"g2p_oov_{name}.dict").read_text(encoding="utf-8")
                    .splitlines() if l.strip()})
    merged = DICTS / f"{name}_plus_g2p.dict"
    shutil.copyfile(path, merged)
    with open(merged, "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(g2p[w] for w in oov) + "\n")
    mfa("align", CORPUS, merged, "english_mfa", OUT / "mfa_ipa_dict_v300", *ALIGN_OPTS, log_name="align_mfa_ipa_dict_v300")
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "ipa_dict_check.py")])


STAGES = {"prepare": stage_prepare, "models": stage_models, "oov": stage_oov, "dict": stage_dict,
          "validate": stage_validate, "align": stage_align, "adapt": stage_adapt, "analyze": stage_analyze,
          "sanity": stage_sanity, "ipa_dict": stage_ipa_dict}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES))
    args = ap.parse_args()
    if shutil.which("mfa") is None:
        raise SystemExit("`mfa` not found - activate the MFA conda env first (see README).")
    for s in args.stages:
        print(f"\n======== stage: {s} ========", flush=True)
        STAGES[s]()


if __name__ == "__main__":
    main()
