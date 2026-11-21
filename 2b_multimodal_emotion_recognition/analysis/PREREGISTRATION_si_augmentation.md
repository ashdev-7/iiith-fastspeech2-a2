# Pre-registration 3: speaker-perturbing augmentation, and regeneration of the noise study

Written and pushed to GitHub **before** either Kaggle job was started. The git server timestamp of this
file is the proof of order. Nothing in this study has been run on the test speaker. The only run so far is
a 1-epoch smoke test of the augmentation code path on the *training* speaker (fold 0, seed 0, with the
original features standing in for the perturbed ones). It saw no test clip and informs no choice.

## Part A: speed-perturbation augmentation (registered as "planned after" in pre-registration 2)

**Question.** Pre-registration 2 found that the BiGRU overfits its single training speaker, and that
per-speaker normalisation does not fix it. Does showing it more "voices" of the same speaker help?

**Augmentation (fixed).**
* Every clip is speed-perturbed at 0.9 and 1.1 with `librosa.resample(y, orig_sr=round(16000*f),
  target_sr=16000)`, so tempo and pitch change together (Kaldi/sox `speed`, as in Ko et al., Interspeech
  2015). The result is peak-normalised to 0.95 like every original clip (`common/audio.py:speed_perturb`).
* WavLM-Base+ features are re-extracted with the unchanged extractor: stores `wavlm_sp090`, `wavlm_sp110`.
* Only **training** rows use the perturbed copies. Validation (early stopping) and test always use the
  original audio. The test speaker's perturbed features are computed but never read.
* Each perturbed copy is its own pseudo-speaker for normalisation. Its statistics come only from that
  copy's clips of that speaker.

**Models (all other settings identical to pre-registration 2).**
* `A-B2`: B2 (BiGRU + speaker norm †, dim 256, dropout 0.2, weight decay 0.01, 40 epochs, patience 8),
  trained on original + 0.9× + 1.1× copies of the training split. Same 10 seeds × 2 directions as B2.
  An epoch now has 3× as many steps. This is a consequence of the design, not a tuned choice.
* `A-L1n`: L1n (pooled mean+std of all 7 layers, per-(pseudo-)speaker z-scoring, StandardScaler, logistic
  regression with C = 0.001 fixed as in study 2), trained on original + both copies of train+val.
* The comparison rows are the existing B2 runs (`runs_capacity/`) and L1n (`capacity_lr_predictions.csv`).
  B2 is not retrained.

**Hypotheses and predictions.**
| id | difference (per direction) | prediction |
|---|---|---|
| HA1 | A-B2 − B2 | > 0 (augmentation helps the BiGRU) |
| HA2 | A-L1n − L1n | ≥ −2 pp (does not hurt the linear head) |
| HA3 | (A-B2 − B2) − (A-L1n − L1n) | > 0 (helps the BiGRU more than the linear head; the prediction registered in study 2) |
| HA4 | A-B2 − L1n | descriptive only: does augmentation close the gap to the linear head? |

**Statistics.**
* **Primary:** a bootstrap over the 200 test words *and* the 10 training seeds (10 000 resamples, seed 0).
  This is the procedure added post hoc in study 2, now registered *before* the data.
  Seeds are paired by seed id for A-B2 vs. B2. L1n and A-L1n are deterministic.
* **Secondary:** the word-only bootstrap of study 2, reported for continuity.
* Verdicts as in study 2 (replicated / direction-specific / equivalent within ±2 pp / inconclusive).
  These use the primary intervals.
* Holm correction over HA1–HA3 × 2 directions (6 tests), using two-sided primary bootstrap p-values.
* Every row is reported whatever the outcome. With two speakers, agreement between the two directions is
  descriptive, not a claim about speakers in general.

## Part N: regenerate the noise-robustness study from retrained checkpoints

The noise table in the report was rebuilt from printed log values, because the original checkpoints were
not kept. Pre-registration 2 showed that retraining reproduces the original runs prediction for prediction.
So the six models of the noise study are retrained with unchanged code and seeds on Kaggle:
speech `wavlm`, `wavlm_spknorm`; text `asr`; fusion `gated_wavlm_gt`, `gated_wavlm_asr`,
`gated_wavlm_asr_spknorm`; SD 5 seeds and SI 6 runs each. `analysis/noise_robustness.py` then runs unchanged.
The job also re-transcribes the clean audio with Whisper from scratch.

**Checks and commitments.**
* **N1:** retrained test predictions are compared with the committed `runs/` predictions for every run.
  The expectation is 100 % identical. Any mismatch is reported, and its source (e.g. a Whisper transcript
  that differs from `artifacts/asr_transcripts.csv`) is traced.
* **N2:** the regenerated noise table **replaces** the log-recovered one in the report, whatever it shows.
  Every cell that differs from the recovered table by more than 0.1 pp is listed.
* `common/engine.py` was refactored for Part A (the training dataset is built before its DataLoader).
  N1 is therefore also the test that this refactor left the existing models unchanged.
