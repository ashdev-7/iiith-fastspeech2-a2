# Pre-registration 4: stronger frozen speech encoders, with leakage controls

Written and pushed to GitHub **before** the Kaggle job was started; the git server timestamp of this file
is the proof of order. The analysis code (`analysis/backbone_features.py`, `analysis/backbone_study.py`)
is committed together with this plan and was validated on random (null) features only.

**What has been run before this plan (full disclosure).** Format checks on a laptop:
* emotion2vec base on 2 clips (feature shape and determinism; no labels involved);
* emotion2vec+ seed (the leakage control below, never a candidate) on 7 clips of speaker YAF, one per
  emotion, with its own classifier head: 6 of 7 matched the TESS label (pleasant surprise → "other").
  This informs no choice: the control's role is fixed below whatever it scores.

## Question

Studies 1–3 found that a low-capacity linear head on frozen WavLM-Base+ features transfers best to the
unseen speaker (A-L1n: 72.0 % †, mean of the two directions). Is the encoder now the bottleneck? Do larger
or emotion-specific frozen encoders, whose training data **documentably excludes TESS**, transfer better?

## Encoders (fixed)

| id | model | training data (as documented by the authors) | features |
|---|---|---|---|
| B | `microsoft/wavlm-base-plus` (reference, = A-L1n of study 3) | 94k h: Libri-Light, GigaSpeech, VoxPopuli; no labels | 7 layers (0, 2, …, 12) |
| WL | `microsoft/wavlm-large` (Chen et al., IEEE JSTSP 2022) | 94k h: Libri-Light, GigaSpeech, VoxPopuli; no labels | 13 layers (0, 2, …, 24) |
| HL | `facebook/hubert-large-ll60k` (Hsu et al., IEEE/ACM TASLP 2021) | 60k h Libri-Light; no labels | 13 layers (0, 2, …, 24) |
| E | `emotion2vec/emotion2vec_base` (Ma et al., Findings of ACL 2024) | LibriSpeech 960 h (init) + 262 h IEMOCAP, MELD, MEAD, CMU-MOSEI, MSP-Podcast; **no emotion labels** ("self-supervised online distillation"); TESS appears nowhere in the paper | final layer (the documented feature) |

None of these corpora contains TESS. The large models use every second hidden state, the same rule as B.
Frames are averaged in pairs (40 ms) and pooled to mean + std per layer, exactly as for B. Each model's own
input normalisation is applied (`AutoFeatureExtractor`; funasr for emotion2vec). All encoders are frozen.

**Excluded from the candidates, by rule.** Any model fine-tuned with emotion labels on data that may contain
TESS. This covers all emotion2vec+ models: the seed model is "fine-tuned with academic speech emotion data
from EmoBox" (model card), and EmoBox lists TESS (Ma et al., Interspeech 2024, Table 1) with a
speaker-dependent split (75 % of the clips for training). The base/large+ models use undisclosed
pseudo-labelled data, so TESS cannot be ruled out. Ready-made "speech emotion recognition" checkpoints
from model hubs are excluded for the same reason.

## Recipe (fixed; identical to A-L1n of study 3)

Pooled mean + std, per-(pseudo-)speaker z-scoring of the pooled vectors (label-free †), StandardScaler,
L2 logistic regression with **C = 0.001** (the value fixed since study 1; no re-tuning per encoder).
Training rows = train + val of the training speaker plus their 0.9× and 1.1× speed-perturbed copies, each
copy normalised as its own pseudo-speaker. Test = original audio of the unseen speaker. The plain variant
without augmentation (L1n) is also reported for each encoder (secondary, descriptive).

## Primary model (fixed before any result)

**ENS** = the equal-weight average of the predicted class probabilities of the four A-L1n models
(B, WL, HL, E), then argmax. The members and weights are fixed here. No encoder is selected by its test
score, so no choice depends on the test speaker.

## Hypotheses

| id | difference (per direction) | prediction |
|---|---|---|
| HB1 (primary) | ENS − A-L1n(B) | > 0 in both directions |
| HB2 | A-L1n(X) − A-L1n(B), X ∈ {WL, HL, E} | none (two-sided; reported for each) |
| HB3 | inductive ENS − transductive ENS on the same 160 words | ≥ −2 pp (the ensemble also works with a 40-word enrolment) |

HB3 uses the 40 calibration words of study 2 (numpy seed 7). The test speaker's normalisation statistics
come only from those clips, for every member, and all models are scored on the other 160 words.

**Reporting rule.** If HB1 is replicated (+), ENS becomes the report's best system. Otherwise A-L1n stays the
best system and ENS is reported as a negative result. Every row is reported whatever it shows.

## Leakage control (never a candidate)

**LC.** emotion2vec+ seed (`emotion2vec/emotion2vec_plus_seed`), which was trained with labels on EmoBox data
that includes TESS.
* LC-a: its own classifier head with no training on our side. Argmax over the 7 classes that map onto
  TESS (angry, disgusted→disgust, fearful→fear, happy, neutral, surprised→ps, sad), accuracy per speaker.
* LC-b: its final-layer features through the same A-L1n recipe.

These are compared descriptively with E (the same architecture and size, without labels). The point is
to show the size of the score that label exposure produces, not to claim anything with it.

## Data and leakage audits (all must pass)

* **D1.** Each speaker says each of the 200 words in all 7 emotions exactly once, so word identity carries
  no label information.
* **D2.** In every direction the training rows contain only the training speaker, and the test rows only
  the test speaker. No clip (uid) appears in both.
* **D3.** Normalisation statistics are computed from features only. The code path that computes them is
  given no labels.
* **D4.** The test speaker's speed-perturbed copies are never read.
* **C1.** A-L1n(B), recomputed in this job, gives 100 % identical predictions to
  `Results/augmentation_lr_predictions.csv`.

## Statistics

* Word-cluster bootstrap over the 200 test words (10 000 resamples, numpy seed 0). All models are
  deterministic, so there is no seed resampling.
* Verdicts as in studies 2–3: replicated / direction-specific / equivalent within ±2 pp / inconclusive.
* Holm correction over HB1 + HB2: 4 comparisons × 2 directions = 8 tests, two-sided bootstrap p-values.
* With two speakers, agreement between the two directions is descriptive, not a claim about speakers in
  general.
* Post hoc, and labelled as such: accuracy against C for each encoder (diagnostic only, never used to
  choose).
