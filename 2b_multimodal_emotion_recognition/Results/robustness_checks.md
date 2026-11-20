# Robustness checks of the statistical claims

## 1. Multiple comparisons (Holm-Bonferroni on the run-level paired t-test)

24 comparisons in `significance.csv`. With n = 6 runs the smallest possible two-sided Wilcoxon p is 0.031, so the Wilcoxon test cannot survive any correction.

| A                      | B                              | protocol   |   runs |   delta_acc_pp |   p_paired_t |   p_wilcoxon |   p_t_holm_within_protocol |   p_t_holm_all |
|:-----------------------|:-------------------------------|:-----------|-------:|---------------:|-------------:|-------------:|---------------------------:|---------------:|
| speech/logmel          | speech/wavlm                   | rand       |      5 |         0.1905 |       0.3375 |       0.5    |                     1      |         1      |
| speech/logmel          | speech/wavlm                   | sd         |      5 |         0.381  |       0.1778 |       0.25   |                     1      |         1      |
| speech/logmel          | speech/wavlm                   | si         |      6 |        16.9048 |       0.0219 |       0.0312 |                     0.1752 |         0.5256 |
| speech/wavlm           | fusion/gated_wavlm_gt          | rand       |      5 |         0.1429 |       0.208  |       0.5    |                     1      |         1      |
| speech/wavlm           | fusion/gated_wavlm_gt          | sd         |      5 |         0.0476 |       0.3739 |       1      |                     1      |         1      |
| speech/wavlm           | fusion/gated_wavlm_gt          | si         |      6 |        -1.7024 |       0.5378 |       0.6875 |                     1      |         1      |
| speech/wavlm           | fusion/gated_wavlm_asr         | rand       |      5 |         0.0476 |       0.7489 |       1      |                     1      |         1      |
| speech/wavlm           | fusion/gated_wavlm_asr         | sd         |      5 |         0.0952 |       0.1778 |       0.5    |                     1      |         1      |
| speech/wavlm           | fusion/gated_wavlm_asr         | si         |      6 |        -1.3333 |       0.7366 |       0.8438 |                     1      |         1      |
| speech/wavlm           | fusion/concat_wavlm_gt         | rand       |      5 |         0      |       1      |       1      |                     1      |         1      |
| speech/wavlm           | fusion/concat_wavlm_gt         | sd         |      5 |         0.0952 |       0.1778 |       0.5    |                     1      |         1      |
| speech/wavlm           | fusion/concat_wavlm_gt         | si         |      6 |         1.6429 |       0.7707 |       0.8438 |                     1      |         1      |
| speech/wavlm           | fusion/xattn_wavlm_gt          | rand       |      5 |         0.1429 |       0.208  |       0.5    |                     1      |         1      |
| speech/wavlm           | fusion/xattn_wavlm_gt          | sd         |      5 |         0      |       1      |       1      |                     1      |         1      |
| speech/wavlm           | fusion/xattn_wavlm_gt          | si         |      6 |         4.9405 |       0.3942 |       0.5625 |                     1      |         1      |
| speech/wavlm           | speech/wavlm_spknorm           | rand       |      5 |         0.1429 |       0.208  |       0.5    |                     1      |         1      |
| speech/wavlm           | speech/wavlm_spknorm           | sd         |      5 |         0.0476 |       0.6213 |       1      |                     1      |         1      |
| speech/wavlm           | speech/wavlm_spknorm           | si         |      6 |        -1.9405 |       0.6881 |       0.5625 |                     1      |         1      |
| fusion/gated_wavlm_asr | fusion/gated_wavlm_asr_spknorm | rand       |      5 |         0.0476 |       0.6213 |       1      |                     1      |         1      |
| fusion/gated_wavlm_asr | fusion/gated_wavlm_asr_spknorm | sd         |      5 |        -0.381  |       0.0777 |       0.125  |                     0.6219 |         1      |
| fusion/gated_wavlm_asr | fusion/gated_wavlm_asr_spknorm | si         |      6 |        -1.7024 |       0.617  |       0.6875 |                     1      |         1      |
| fusion/gated_wavlm_gt  | fusion/gated_wavlm_asr         | rand       |      5 |        -0.0952 |       0.1778 |       0.5    |                     1      |         1      |
| fusion/gated_wavlm_gt  | fusion/gated_wavlm_asr         | sd         |      5 |         0.0476 |       0.3739 |       1      |                     1      |         1      |
| fusion/gated_wavlm_gt  | fusion/gated_wavlm_asr         | si         |      6 |         0.369  |       0.9119 |       0.8438 |                     1      |         1      |

## 2. SI comparisons per transfer direction

Mean difference B - A over the 3 seeds of each direction (pp), with [min, max] over seeds. The 6 SI runs are not 6 independent samples: seeds within a direction share the training speaker and the test speaker. A difference that changes sign between directions is not established by this corpus.

| A             | B                      | delta OAF->YAF (3 seeds)   | delta YAF->OAF (3 seeds)   | same sign both directions   |
|:--------------|:-----------------------|:---------------------------|:---------------------------|:----------------------------|
| speech/logmel | speech/wavlm           | +9.57 [+2.4, +22.4]        | +24.24 [+14.0, +35.1]      | yes                         |
| speech/wavlm  | fusion/gated_wavlm_gt  | -2.00 [-7.4, +1.1]         | -1.40 [-10.7, +6.8]        | yes                         |
| speech/wavlm  | fusion/gated_wavlm_asr | -1.62 [-8.1, +6.2]         | -1.05 [-14.9, +9.6]        | yes                         |
| speech/wavlm  | fusion/concat_wavlm_gt | +1.00 [-17.7, +13.6]       | +2.29 [-11.6, +12.1]       | yes                         |
| speech/wavlm  | fusion/xattn_wavlm_gt  | +13.02 [+0.9, +28.1]       | -3.14 [-7.1, +3.5]         | no                          |
| speech/wavlm  | speech/wavlm_spknorm   | +3.24 [-5.1, +14.2]        | -7.12 [-20.4, -0.1]        | no                          |

## 3. Regression-to-the-mean control for 'fusion helps when speech is uncertain'

Clips are binned by the confidence of the speech model (seed s). `fusion - speech` is the gain of the fusion model of the same seed (the figure in the report). `other speech seed - speech` is the same computation with a *second speech-only model* (another seed, same direction) in place of fusion: any gain it shows is a pure selection artefact. Only the net effect can be attributed to fusion.

|                                |    n |   fusion - speech (pp) |   other speech seed - speech (pp) |   net fusion effect (pp) |
|:-------------------------------|-----:|-----------------------:|----------------------------------:|-------------------------:|
| ('gated_wavlm_asr', '<0.5')    | 1002 |                  11.78 |                              6.09 |                     5.69 |
| ('gated_wavlm_asr', '0.5-0.7') | 1427 |                   5.26 |                              3.89 |                     1.37 |
| ('gated_wavlm_asr', '0.7-0.9') | 2497 |                  -0.52 |                              1.4  |                    -1.92 |
| ('gated_wavlm_asr', '>=0.9')   | 3474 |                  -8.41 |                             -4.36 |                    -4.04 |
| ('gated_wavlm_gt', '<0.5')     | 1002 |                   4.89 |                              6.09 |                    -1.2  |
| ('gated_wavlm_gt', '0.5-0.7')  | 1427 |                   1.12 |                              3.89 |                    -2.77 |
| ('gated_wavlm_gt', '0.7-0.9')  | 2497 |                  -3.16 |                              1.4  |                    -4.57 |
| ('gated_wavlm_gt', '>=0.9')    | 3474 |                  -3.71 |                             -4.36 |                     0.65 |
