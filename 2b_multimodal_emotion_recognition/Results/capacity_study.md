# Pre-registered head-capacity study (SI)

Plan: `analysis/PREREGISTRATION_si_capacity.md` (committed before the runs). B1 = BiGRU (main), B2 = BiGRU + speaker norm †, B3 = small BiGRU + norm †, B4 = attentive statistics pooling + norm †, L1/L1n = logistic regression (mean+std) without/with norm †, L2n = logistic regression (mean only) + norm †. † transductive.

## Accuracy per direction (10 seeds for neural heads)

| direction   | model   | run_dir                        |   seeds |   mean |     sd |    min |    max |   ensemble |
|:------------|:--------|:-------------------------------|--------:|-------:|-------:|-------:|-------:|-----------:|
| OAF->YAF    | B1      | wavlm                          |      10 |  57.18 |  10.46 |  44.93 |  72.14 |      61.79 |
| OAF->YAF    | B2      | wavlm_spknorm                  |      10 |  57.26 |  10.72 |  43.5  |  72.79 |      62.07 |
| OAF->YAF    | B3      | wavlm_spknorm_d64_do0.5_wd0.05 |      10 |  61.44 |   5.29 |  52.57 |  70.29 |      69    |
| OAF->YAF    | B4      | wavlm_spknorm_attnpool         |      10 |  58.23 |   9.34 |  43    |  73.5  |      58.36 |
| OAF->YAF    | L1      | logistic regression            |       1 |  68.43 | nan    | nan    | nan    |     nan    |
| OAF->YAF    | L1n     | logistic regression            |       1 |  75.29 | nan    | nan    | nan    |     nan    |
| OAF->YAF    | L2n     | logistic regression            |       1 |  73.14 | nan    | nan    | nan    |     nan    |
| YAF->OAF    | B1      | wavlm                          |      10 |  61.45 |   5.63 |  53.64 |  74.71 |      64    |
| YAF->OAF    | B2      | wavlm_spknorm                  |      10 |  55.71 |   5.3  |  48.64 |  62.57 |      56.79 |
| YAF->OAF    | B3      | wavlm_spknorm_d64_do0.5_wd0.05 |      10 |  60.26 |   5.45 |  52.64 |  70.86 |      63.64 |
| YAF->OAF    | B4      | wavlm_spknorm_attnpool         |      10 |  55.56 |   3.85 |  51.14 |  64    |      55.36 |
| YAF->OAF    | L1      | logistic regression            |       1 |  63.5  | nan    | nan    | nan    |     nan    |
| YAF->OAF    | L1n     | logistic regression            |       1 |  66    | nan    | nan    | nan    |     nan    |
| YAF->OAF    | L2n     | logistic regression            |       1 |  62.29 | nan    | nan    | nan    |     nan    |

## Pre-registered comparisons (word-cluster bootstrap, 10 000 resamples; ROPE +-2 pp; Holm over all 14 tests)

| hypothesis        | comparison   | direction   |   delta_pp |   ci_lo |   ci_hi |   p_boot | prediction    |   ci2_lo_posthoc |   ci2_hi_posthoc | verdict             | prediction_supported   |   p_holm |
|:------------------|:-------------|:------------|-----------:|--------:|--------:|---------:|:--------------|-----------------:|-----------------:|:--------------------|:-----------------------|---------:|
| H1 (partly blind) | B2 - B1      | OAF->YAF    |       0.09 |   -0.55 |    0.74 |   0.8212 | > 0           |            -8.79 |             8.16 | direction-specific  | no                     |   0.8212 |
| H1 (partly blind) | B2 - B1      | YAF->OAF    |      -5.74 |   -6.4  |   -5.09 |   0      | > 0           |           -10.31 |            -1.66 | direction-specific  | no                     |   0      |
| H2                | L1n - B2     | OAF->YAF    |      18.02 |   16.51 |   19.52 |   0      | >= -2 pp      |            11.55 |            24.69 | replicated (+)      | yes                    |   0      |
| H2                | L1n - B2     | YAF->OAF    |      10.29 |    9.18 |   11.41 |   0      | >= -2 pp      |             6.97 |            13.57 | replicated (+)      | yes                    |   0      |
| H3                | B3 - B2      | OAF->YAF    |       4.18 |    3.34 |    5.01 |   0      | > 0           |            -2.3  |            11.34 | replicated (+)      | yes                    |   0      |
| H3                | B3 - B2      | YAF->OAF    |       4.54 |    3.93 |    5.14 |   0      | > 0           |            -0.89 |             9.86 | replicated (+)      | yes                    |   0      |
| H4                | B4 - B3      | OAF->YAF    |      -3.21 |   -4.27 |   -2.15 |   0      | >= -2 pp      |            -8.53 |             2.31 | replicated (-)      | no                     |   0      |
| H4                | B4 - B3      | YAF->OAF    |      -4.69 |   -5.66 |   -3.73 |   0      | >= -2 pp      |            -9.36 |            -0.55 | replicated (-)      | no                     |   0      |
| H5                | E(B2) - B2   | OAF->YAF    |       4.81 |    3.54 |    6.06 |   0      | > 0           |           nan    |           nan    | replicated (+)      | yes                    |   0      |
| H5                | E(B2) - B2   | YAF->OAF    |       1.07 |    0.12 |    1.99 |   0.0268 | > 0           |           nan    |           nan    | replicated (+)      | yes                    |   0.0804 |
| H6                | I(L1n) - L1n | OAF->YAF    |      -0.36 |   -0.8  |    0    |   0.1354 | within +-2 pp |           nan    |           nan    | equivalent (+-2 pp) | yes                    |   0.2708 |
| H6                | I(L1n) - L1n | YAF->OAF    |      -0.62 |   -1.07 |   -0.18 |   0.0014 | within +-2 pp |           nan    |           nan    | equivalent (+-2 pp) | yes                    |   0.0056 |
| H7                | L2n - L1n    | OAF->YAF    |      -2.14 |   -3.14 |   -1.21 |   0      | <= 0          |            -3.14 |            -1.21 | replicated (-)      | yes                    |   0      |
| H7                | L2n - L1n    | YAF->OAF    |      -3.71 |   -4.71 |   -2.79 |   0      | <= 0          |            -4.71 |            -2.79 | replicated (-)      | yes                    |   0      |

`ci_lo/ci_hi` = the pre-registered word bootstrap. It treats the 10 seeds as fixed, so it measures uncertainty over test words only. `ci2_*_posthoc` = **post-hoc** (not pre-registered) bootstrap over words AND seeds, which also includes training noise; it is the more honest interval for a claim about the model rather than these 10 trained instances.

## Inductive normalisation of the BiGRU (B2): 40 calibration words, scored on the other 160

| direction | transductive | inductive | delta | seeds |
|---|---|---|---|---|
| OAF->YAF | 56.91 | 56.81 | -0.10 | 10 |
| YAF->OAF | 55.71 | 55.99 | +0.28 | 10 |

## Reproducibility: retrained seeds 0-2 vs. the original runs

| model   | run            |   original |   retrained |   same_prediction_pct |
|:--------|:---------------|-----------:|------------:|----------------------:|
| B1      | si_fold0_seed0 |      52.14 |       52.14 |                   100 |
| B1      | si_fold0_seed1 |      51.07 |       51.07 |                   100 |
| B1      | si_fold0_seed2 |      55.86 |       55.86 |                   100 |
| B1      | si_fold1_seed0 |      61.43 |       61.43 |                   100 |
| B1      | si_fold1_seed1 |      74.71 |       74.71 |                   100 |
| B1      | si_fold1_seed2 |      55.64 |       55.64 |                   100 |
| B2      | si_fold0_seed0 |      52.71 |       52.71 |                   100 |
| B2      | si_fold0_seed1 |      65.29 |       65.29 |                   100 |
| B2      | si_fold0_seed2 |      50.79 |       50.79 |                   100 |
| B2      | si_fold1_seed0 |      61.36 |       61.36 |                   100 |
| B2      | si_fold1_seed1 |      54.29 |       54.29 |                   100 |
| B2      | si_fold1_seed2 |      54.79 |       54.79 |                   100 |
