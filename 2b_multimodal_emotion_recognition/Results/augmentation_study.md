# Pre-registered augmentation study (SI)

Plan: `analysis/PREREGISTRATION_si_augmentation.md` (pushed before the runs). A- = trained on the training speaker's original clips plus 0.9x and 1.1x speed-perturbed copies (each copy normalised as its own pseudo-speaker). † transductive normalisation in every row.

## Accuracy per direction

| direction   | model   |   seeds |   mean |     sd |    min |    max |   ensemble |   l1n_identical_to_study2_pct |
|:------------|:--------|--------:|-------:|-------:|-------:|-------:|-----------:|------------------------------:|
| OAF->YAF    | B2      |      10 |  57.26 |  10.72 |  43.5  |  72.79 |      62.07 |                           nan |
| OAF->YAF    | A-B2    |      10 |  54.48 |   5.8  |  45    |  63.93 |      56.36 |                           nan |
| OAF->YAF    | L1n     |       1 |  75.29 | nan    | nan    | nan    |     nan    |                           100 |
| OAF->YAF    | A-L1n   |       1 |  76.36 | nan    | nan    | nan    |     nan    |                           nan |
| YAF->OAF    | B2      |      10 |  55.71 |   5.3  |  48.64 |  62.57 |      56.79 |                           nan |
| YAF->OAF    | A-B2    |      10 |  59.37 |   3.53 |  53.5  |  65.29 |      58.43 |                           nan |
| YAF->OAF    | L1n     |       1 |  66    | nan    | nan    | nan    |     nan    |                           100 |
| YAF->OAF    | A-L1n   |       1 |  67.71 | nan    | nan    | nan    |     nan    |                           nan |

## Pre-registered tests

Primary CI = bootstrap over test words and seeds (10 000 resamples); `*_words_only` = the word-only bootstrap of study 2 (secondary). Holm over HA1-HA3 x 2 directions. Verdicts use the primary CIs with a ROPE of +-2 pp.

| hypothesis   | comparison                  | direction   |   delta_pp |   ci_lo |   ci_hi |   p_boot | prediction   |   ci_lo_words_only |   ci_hi_words_only |   p_holm | verdict        | prediction_supported   |
|:-------------|:----------------------------|:------------|-----------:|--------:|--------:|---------:|:-------------|-------------------:|-------------------:|---------:|:---------------|:-----------------------|
| HA1          | A-B2 - B2                   | OAF->YAF    |      -2.79 |   -9.62 |    3.84 |   0.4222 | > 0          |              -3.46 |              -2.09 |   0.7836 | inconclusive   | no                     |
| HA2          | A-L1n - L1n                 | OAF->YAF    |       1.07 |    0.57 |    1.64 |   0      | >= -2 pp     |               0.57 |               1.64 |   0      | replicated (+) | yes                    |
| HA3          | (A-B2 - B2) - (A-L1n - L1n) | OAF->YAF    |      -3.86 |  -10.69 |    2.78 |   0.2612 | > 0          |              -4.79 |              -2.94 |   0.7836 | inconclusive   | no                     |
| HA4          | A-B2 - L1n (descriptive)    | OAF->YAF    |     -20.81 |  -24.57 |  -17.19 |   0      | none         |             -22.07 |             -19.58 | nan      | replicated (-) | -                      |
| HA1          | A-B2 - B2                   | YAF->OAF    |       3.66 |   -1.01 |    7.72 |   0.1124 | > 0          |               3.18 |               4.14 |   0.4496 | inconclusive   | no                     |
| HA2          | A-L1n - L1n                 | YAF->OAF    |       1.71 |    1    |    2.5  |   0      | >= -2 pp     |               1    |               2.5  |   0      | replicated (+) | yes                    |
| HA3          | (A-B2 - B2) - (A-L1n - L1n) | YAF->OAF    |       1.94 |   -2.8  |    6.07 |   0.384  | > 0          |               1.11 |               2.74 |   0.7836 | inconclusive   | no                     |
| HA4          | A-B2 - L1n (descriptive)    | YAF->OAF    |      -6.63 |   -9.11 |   -4.27 |   0      | none         |              -7.79 |              -5.5  | nan      | replicated (-) | -                      |
