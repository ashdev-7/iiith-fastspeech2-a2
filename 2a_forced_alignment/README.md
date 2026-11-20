# Forced Alignment with the Montreal Forced Aligner (MFA)

Word- and phone-level forced alignment of the provided 6-utterance dataset
(3 × BU Radio News read speech, speaker F2B, 25–31 s each; 3 × ISLE L2-English
learner minimal-pair prompts, ~4 s each) with **MFA 3.4.4**. The pipeline is fully
automated, and the repo compares **3 acoustic-model / dictionary configurations
plus a speaker-adapted acoustic model**.

| Experiment | Acoustic model | Pronunciation dictionary |
|---|---|---|
| `arpa_base` (primary) | `english_us_arpa` v3.0.0 (ARPAbet) | `english_us_arpa` v3.0.0 + G2P for the 4 OOVs |
| `mfa_ipa` | `english_mfa` v3.1.0 (IPA phone set) | `english_mfa` v3.1.0 + `english_us_mfa` G2P for the 5 OOVs |
| `arpa_g2p_custom` | `english_us_arpa` | **custom dictionary generated from the transcripts** with the `english_us_arpa` G2P model (2-best prons/word, no lexicon lookup) |
| `arpa_adapted` | `english_us_arpa` **MAP-adapted on this corpus** (`mfa adapt`) | as `arpa_base` |

Outputs: `outputs/<experiment>/<speaker>/<utterance>.TextGrid` (tiers `words`, `phones`).
Report: [`report/report.pdf`](report/report.pdf).

---

## 1. Install MFA

MFA depends on Kaldi binaries (`kalpy`) that are only distributed through conda-forge.

```bash
# any conda flavour works: Miniforge, Miniconda, Anaconda or a portable micromamba
conda env create -f environment.yml          # or:  conda create -n mfa -c conda-forge montreal-forced-aligner=3.4.4
conda activate mfa
mfa version                                  # -> 3.4.4
```

> Used here on Windows 11 with portable **micromamba** (`micromamba create -n mfa -c conda-forge montreal-forced-aligner python=3.11`)
> and `micromamba run -n mfa <command>`. MFA stores models in `MFA_ROOT_DIR` (default `~/Documents/MFA`); set
> `MFA_ROOT_DIR` to a non-synced folder if your Documents folder is on OneDrive.

### Download pretrained models

```bash
mfa model download acoustic   english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p        english_us_arpa
mfa model download acoustic   english_mfa
mfa model download dictionary english_mfa
mfa model download g2p        english_us_mfa     # note: the English IPA G2P model is called english_us_mfa
```

`mfa model download` uses the GitHub API (60 unauthenticated requests/hour, and each call pages through the
whole model list). If you hit the limit, pass `--github_token`, or download the files from the
[mfa-models releases](https://github.com/MontrealCorpusTools/mfa-models/releases) into
`$MFA_ROOT_DIR/pretrained_models/{acoustic,dictionary,g2p}/`. `run_pipeline.py` skips models that are already present.

## 2. Prepare the dataset

```bash
python scripts/prepare_corpus.py --raw data/raw --out corpus
```

MFA wants one folder per speaker, with a `.lab` (or `.txt`) transcript next to each `.wav`:

```
corpus/
├── f2b/             F2BJRLP1.wav  F2BJRLP1.lab  … (BU Radio News, speaker f2b)
└── isle_sess0131/   ISLE_SESS0131_BLOCKD02_01_sprt1.wav  …01_sprt1.lab  …
```

What the script does, and why:

* **Audio container.** The three `F2B*.wav` files are **NIST SPHERE** files (`NIST_1A` header,
  BU Radio Corpus) despite their `.wav` extension. MFA's audio reader handles them as-is, so this is not
  strictly required; they are rewritten as standard RIFF/WAVE (for tools that trust the extension) with every
  sample unchanged (verified by the `sanity` stage). All files are 16 kHz / mono / 16-bit, so there is no resampling.
* **Text normalisation**, logged in `corpus/normalization_log.tsv`:
  `seventy- six → seventy six` (numbers split across line breaks), `S.J.C.'s → s j c's`,
  `WBUR's → w b u r's` (acronyms read letter by letter), `de-politicize → depoliticize`, punctuation
  removed, lower-cased.

## 3. Dictionaries: OOVs, G2P and a custom dictionary

```bash
# out-of-vocabulary words (clitic-aware: "state's" counts as covered when "state" and "'s" are)
python scripts/run_pipeline.py --stages oov
# G2P for the OOVs, merged into the pretrained dictionary
mfa g2p dictionaries/oov_english_us_arpa.txt english_us_arpa dictionaries/g2p_oov_english_us_arpa.dict
# a custom dictionary generated entirely from the transcripts (2 pronunciations per word)
mfa g2p dictionaries/corpus_wordlist.txt english_us_arpa dictionaries/g2p_custom_arpa.dict --num_pronunciations 2
```

| Dictionary | OOV words | G2P pronunciation |
|---|---|---|
| english_us_arpa | dukakis, maffy, melnicove, depoliticize | `D UW0 K AA1 K IH0 S`, `M AE1 F IY0`, `M EH1 L N IH0 K OW2 V`, `D EH2 P AH0 L IH1 T IH0 S AY2 Z` |
| english_mfa | the same four, plus judgeships | `d ɐ cʰ æ c ɪ s`, `m æ fʲ i`, `m ɛ ɫ ɲ ɪ kʰ ow v`, … |

## 4. Run forced alignment

```bash
mfa validate corpus dictionaries/english_us_arpa_plus_g2p.dict --acoustic_model_path english_us_arpa
mfa align corpus dictionaries/english_us_arpa_plus_g2p.dict english_us_arpa outputs/arpa_base \
          --clean --beam 100 --retry_beam 400
mfa align corpus dictionaries/english_mfa_plus_g2p.dict english_mfa outputs/mfa_ipa --clean --beam 100 --retry_beam 400
mfa align corpus dictionaries/g2p_custom_arpa.dict english_us_arpa outputs/arpa_g2p_custom --clean --beam 100 --retry_beam 400
mfa adapt corpus dictionaries/english_us_arpa_plus_g2p.dict english_us_arpa models/english_us_arpa_adapted.zip \
          --output_directory outputs/arpa_adapted --clean --beam 100 --retry_beam 400
```

The wider beam (100 / retry 400, against the default 10 / 40) is used because the BU Radio utterances
are 25–31 s long; with narrow beams, long utterances are the usual cause of "could not align" failures.

### Or run everything with one command

```bash
python scripts/run_pipeline.py            # prepare → models → oov → dict → validate → align → adapt → analyze → sanity
python scripts/run_pipeline.py --stages align analyze
python scripts/run_pipeline.py --stages sanity        # data-preparation and repeatability checks only
```

The `sanity` stage checks the data preparation and MFA itself:
* **audio identity**: every corpus `.wav` carries exactly the provided samples (the three `F2B*.wav` files are
  NIST SPHERE; libsndfile/MFA can read them as-is, and we rewrite them as RIFF without changing a sample);
* **untouched input**: the `arpa_base` setup run on the provided files exactly as given (`corpus_untouched/` →
  `outputs/arpa_untouched/`). Word boundaries are identical to `arpa_base` for every word whose text we did not edit,
  up to MFA's run-to-run noise. Only `WBUR's` and `de-politicize` differ: untouched, they become `spn` (no phones);
* **repeatability**: `arpa_base` re-run twice (`outputs/repeatability/run{1,2}/`): ≤10 ms word / ≤20 ms phone
  differences between identical runs, which is the noise floor against which every comparison is judged.

Results: `analysis/sanity_checks.md` (+ `sanity_*.csv`).

The `ipa_dict` stage (`--stages ipa_dict`) re-aligns with the `english_mfa` acoustic model and the **full v3.0.0**
`english_mfa` dictionary (all variants, with probabilities) instead of MFA's default v3.1.0 dictionary, which keeps a
single pronunciation per word (e.g. *said* → `s aj d`). The download is verified against the release's MD5.
Result (`analysis/ipa_dict_check.md`): the pronunciations become correct (`s ɛ d`), but the silence absorbed into
*bait*/*bed* on the ISLE prompts is unchanged, so that error comes from the acoustic model, not the lexicon; on the
news speech the full dictionary moves `english_mfa` closer to `english_us_arpa` (95.2 % vs 88.2 % of word boundaries within 20 ms).

Each `mfa` call is logged to `logs/<stage>.log`.

## 5. Inspect the output

```bash
python scripts/analyze_alignments.py      # tables in analysis/, figures in analysis/figures/
```

* **Praat (interactive)**: open `corpus/<spk>/<utt>.wav` and `outputs/<exp>/<spk>/<utt>.TextGrid`, select both, then click
  *View & Edit*; click an interval and press Tab to play it. To render a figure from Praat in batch mode:
  `Praat.exe --run scripts/praat/draw_alignment.praat corpus/f2b/F2BJRLP1.wav outputs/arpa_base/f2b/F2BJRLP1.TextGrid analysis/figures/praat_F2BJRLP1.png 0 6`
* **Praat (headless, no GUI needed)**: `pip install praat-parselmouth` then `python scripts/praat/run_praat.py`. This runs
  the same Praat script inside Praat 6.1 (embedded in parselmouth) and writes `analysis/figures/praat_*.pdf`
  (headless Praat exports vector graphics: EPS, converted with `epstopdf`), plus Praat intensity/voicing
  measurements of every stretch where `english_mfa` and `english_us_arpa` disagree (`analysis/praat_measurements.csv`).
* `analysis/figures/align_<exp>_<utt>.png`: waveform + spectrogram + word/phone tiers (Praat-style).
* `analysis/figures/overlay_<utt>_*.png`: word boundaries from all four experiments on one spectrogram.
* `analysis/summary_by_experiment.csv`, `agreement_vs_arpa_base.csv`, `pronunciation_diffs.csv`,
  `energy_flags.csv`: see the report for a discussion.

## 6. Checking accuracy, not just agreement

Agreement between aligners measures consistency, not accuracy. Two references that are independent of every aligner:

* **Aligner-free pause reference** (`python scripts/reference_check.py`, needs parselmouth). It marks pauses as stretches of
  ≥150 ms within 6, 9 or 12 dB of each file's noise floor in Praat's intensity contour. On ISLE the ARPA models put ≤1 % of
  that pause time inside words at every threshold, and `english_mfa` puts 35–56 % there. This confirms the pause-absorption error.
  Edge positions move with the threshold, so this reference is too coarse to rank aligners at 10–20 ms
  (`analysis/reference_check.md`).
* **Blind hand-labelled reference** (`gold/ANNOTATION_PROTOCOL.md`). It covers 27 words (54 boundaries): all ISLE words plus 12
  random news words. Each is marked twice in Praat, blind to every aligner output. `python scripts/score_gold.py` then reports
  the annotator's own pass-to-pass consistency (the reference's precision) and each experiment's error against it.
  The kit (`gold/kit/`) is built by `scripts/make_gold_kit.py`.

## Repository layout

```
data/raw/{wav,transcripts}   provided dataset (unchanged)
corpus/                      MFA-formatted corpus + normalization_log.tsv
dictionaries/                OOV lists, G2P output, merged and custom dictionaries
outputs/<experiment>/        TextGrids (+ MFA's alignment_analysis.csv)
outputs/arpa_untouched/      sanity check: alignment of the untouched provided files
outputs/repeatability/       sanity check: identical re-runs of arpa_base
outputs/mfa_ipa_dict_v300/   english_mfa acoustic model + full v3.0.0 dictionary (ipa_dict stage)
corpus_untouched/            the provided files as given, in MFA's folder layout (built by the sanity stage)
models/                      speaker-adapted acoustic model (git-ignored, 88 MB; rebuilt by the adapt stage)
analysis/                    CSV tables and figures (praat_*.pdf are drawn by Praat itself)
scripts/                     prepare_corpus.py, run_pipeline.py, analyze_alignments.py, sanity_checks.py,
                             ipa_dict_check.py, reference_check.py, make_gold_kit.py, score_gold.py,
                             praat/draw_alignment.praat, praat/run_praat.py
gold/                        blind hand-labelling protocol and kit (two passes)
report/                      report.tex / report.pdf
logs/                        one log per MFA command
```
