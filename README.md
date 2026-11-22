# Assignment 2: Speech Processing

Both parts of Assignment 2. Each folder is self-contained, with its own README (setup and commands), code, outputs and report.

| Part | Folder | Task | Report |
|---|---|---|---|
| **2a** | [`2a_forced_alignment/`](2a_forced_alignment) | Forced alignment with the Montreal Forced Aligner: data preparation, pretrained and custom (G2P) dictionaries, three acoustic models plus speaker adaptation, TextGrid outputs, Praat inspection, error analysis | [`report/report.pdf`](2a_forced_alignment/report/report.pdf) |
| **2b** | [`2b_multimodal_emotion_recognition/`](2b_multimodal_emotion_recognition) | Emotion recognition on TESS with speech-only, text-only and multimodal (fusion) pipelines; 176 training runs of the 11 variants under three evaluation protocols (324 with the ablation and study runs); analysis of easy and hard emotions, when fusion helps, failure cases and cluster separability | [`report/report.pdf`](2b_multimodal_emotion_recognition/report/report.pdf) |

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
| Pipeline | Random split | Seen speakers | Unseen speaker |
|---|---|---|---|
| Speech (WavLM + BiGRU) | 99.86 % | 99.90 % | 58.48 % |
| Text (DistilRoBERTa + BiGRU) | 13.81 % | 14.29 % | 14.29 % |
| Fusion (gated, speech + text) | 100.00 % | 99.95 % | 56.77 % |

- Speech carries the emotion. Text is at chance by construction: in TESS every word is spoken in all seven emotions.
- Fusion therefore matches speech but does not beat it; it helps only where speech is uncertain and the text comes from Whisper, whose punctuation carries prosody.
- The real difficulty is a new speaker (from 99.9 % to 58.5 %). Five additional studies, each planned before it was run, raise this to 75.7 % with frozen WavLM-Large, and show that encoders trained on data containing TESS score 99.9 % on the "unseen" speaker, so they are excluded.
