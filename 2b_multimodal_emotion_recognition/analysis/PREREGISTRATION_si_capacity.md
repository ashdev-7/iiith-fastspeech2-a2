# Pre-registration 2: head capacity, seeds and normalisation under SI

Committed before any run of the new variants. The design follows the literature review in
`../../reports/Speaker independent SER architecture.md` (not part of the repo), in particular Pepino et al.
(Interspeech 2021): an LSTM head on frozen SSL features was no better than a dense head and much less
stable across seeds.

## What is already known (so not a prediction)
* The SI results of pre-registration 1 (`PREREGISTRATION_si_linear.md`, `Results/si_linear.md`) are known.
* The SI runs of WavLM + BiGRU, with and without speaker normalisation, seeds 0–2, are known
  (58.5 % and 56.5 %). Their seeds 3–9 are new. Hypothesis H1 below is therefore only partly blind and is
  marked as such.

## Variants (all frozen WavLM-Base+, 7 layers; SI protocol, both directions)
Neural heads use the existing training loop unchanged: AdamW, one-cycle, ≤40 epochs, early stopping on the
**training speaker's** held-out words. **10 seeds (0–9) per direction.** No hyper-parameter is tuned.

| id | head | normalisation | capacity setting (fixed now) |
|---|---|---|---|
| B1 | layer-weighted projection → 2-layer BiGRU → attention (main model) | none | dim 256, dropout 0.2, wd 0.01 |
| B2 | same | per-speaker † | same |
| B3 | same, "small" | per-speaker † | dim 64, dropout 0.5, wd 0.05 |
| B4 | layer-weighted sum → LayerNorm → attentive statistics pooling (weighted mean+std) → linear; no recurrence | per-speaker † | dropout 0.2 on the input, wd 0.01 |
| L1 | pre-registration 1: pooled mean+std, all layers, L2 logistic regression | none | as registered |
| L1n | same | per-speaker † | as registered |
| L2n | pooled **mean only**, all layers, L2 logistic regression | per-speaker † | C = 0.001 (L1's value, fixed now) |

† label-free per-speaker standardisation using the test speaker's unlabelled audio (transductive).

**Inductive normalisation check (I).** 40 of the 200 words are drawn once (numpy seed 7, before scoring) as
a calibration set. The test speaker's statistics are computed from the 280 calibration clips only; all
compared models are then scored on the remaining 160 words (1120 clips). This is compared with the
transductive statistics on the same 1120 clips, for L1n and B2.

**Seed ensemble (E).** For B1–B4, the posterior averaged over the 10 seeds, per direction.

## Hypotheses and pre-declared comparisons (Holm-corrected as one family)
| # | comparison (per direction) | prediction |
|---|---|---|
| H1 (partly blind) | B2 − B1 | normalisation helps the BiGRU (> 0) |
| H2 | L1n − B2 | the linear probe is at least as good as the main BiGRU (≥ −2 pp in both directions) |
| H3 | B3 − B2 | less capacity is better (> 0) |
| H4 | B4 − B3 | pooling without recurrence is at least as good (≥ −2 pp) |
| H5 | E(B2) − mean of B2 seeds | the seed ensemble is better (> 0) |
| H6 | I(L1n) − L1n, same 1120 clips | inductive normalisation keeps most of the gain: within ±2 pp |
| H7 | L2n − L1n | dropping the std statistics does not help (≤ 0) |

## Analysis plan
* Metric: accuracy (= UAR, classes balanced) and macro-F1.
* For seeded models, the per-direction score is the mean over the 10 seeds; seed SD and range are reported.
* Uncertainty: **word-cluster bootstrap**, 10 000 resamples of the 200 test words (each word carries its 7
  clips and all seeds), numpy seed 0, giving a 95 % percentile CI for every paired difference, per direction.
* Verdict per comparison, with a **±2 pp equivalence band (ROPE)**, a convention fixed now:
  * *replicated*: the CI excludes 0 and is on the same side in both directions;
  * *direction-specific*: that holds in one direction only;
  * *equivalent*: the CI lies inside ±2 pp in both directions;
  * *inconclusive*: otherwise.
* Holm correction over H1–H7 × 2 directions, using bootstrap two-sided p-values.
* No claim about speakers in general: with 2 speakers, agreement across directions is descriptive.
* Every variant listed here is reported, whatever the outcome.

## Planned after this, registered now
**Speaker-perturbing augmentation (A).**
* Training audio of the training speaker is speed-perturbed at 0.9 and 1.1 (resampling, so pitch and
  tempo change together). WavLM features are re-extracted.
* Each perturbed copy is treated as its own pseudo-speaker for normalisation.
* Models: B2 and L1n, trained on original + both copies. Prediction: A helps B2 more than L1n.
