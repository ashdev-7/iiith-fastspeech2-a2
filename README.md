# Assignment 2 — Speech Processing

Both parts of Assignment 2. Each folder is self-contained, with its own README (setup and commands), code, outputs and report.

| Part | Folder | Task | Report |
|---|---|---|---|
| **2a** | [`2a_forced_alignment/`](2a_forced_alignment) | Forced alignment with the Montreal Forced Aligner: data preparation, pretrained and custom (G2P) dictionaries, three acoustic models plus speaker adaptation, TextGrid outputs, Praat inspection, error analysis | [`report/report.pdf`](2a_forced_alignment/report/report.pdf) |
| **2b** | [`2b_multimodal_emotion_recognition/`](2b_multimodal_emotion_recognition) | Emotion recognition on TESS with speech-only, text-only and multimodal (fusion) pipelines; 224 training runs under three evaluation protocols; analysis of easy and hard emotions, when fusion helps, failure cases and cluster separability | [`report/report.pdf`](2b_multimodal_emotion_recognition/report/report.pdf) |

## Quick start

```bash
# 2a: MFA pipeline (conda/micromamba env from environment.yml)
cd 2a_forced_alignment
conda env create -f environment.yml && conda activate mfa
python scripts/run_pipeline.py

# 2b: emotion recognition (GPU recommended; every reported number was produced on a Kaggle T4)
cd 2b_multimodal_emotion_recognition
pip install -r requirements.txt
python prepare_data.py --all && python run_all.py
```

## Headline results

**2a.**
- `english_us_arpa` with its curated dictionary plus G2P for OOVs gives the most reliable alignment.
- The `english_mfa` model absorbs pauses into words on the short learner prompts. Praat measures those stretches at 24–26 dB below speech and unvoiced.
- This error is *not* caused by its dictionary: the full dictionary fixes the pronunciations but not the pauses.
- Run-to-run variation in MFA (≤10 ms) is measured and used as the noise floor for every comparison.

**2b.**
- Speech-only WavLM + BiGRU reaches 99.9 % with the speaker seen in training, but 58.5 % on an unseen speaker: it overfits the one training speaker.
- A pre-registered low-capacity model (plan committed before the run) raises unseen-speaker accuracy to 66.0 % (70.7 % with label-free speaker normalisation).
- Text with ground-truth transcripts is at chance by construction: every word appears with every emotion.
- Whisper transcripts reach 2.1× chance, because ASR punctuation carries prosody.
- Fusion is not significant on average. Where speech is uncertain, Whisper-text fusion gains +5.7 pp after a regression-to-the-mean control removes half of the raw +11.8 pp.
- Statistics are conservative: Holm correction over 24 comparisons, and per-direction checks, because 6 SI runs come from only 2 speakers.
