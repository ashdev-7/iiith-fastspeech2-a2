# When does fusion help?

## Paired comparison against the speech-only model (same split & seed)

`fixed` = clips the speech model got wrong and fusion got right; `broken` = the reverse. `mean_gate_speech` = mean GMU gate z (1 = rely fully on speech).

| fusion          | protocol   |   speech_acc |   fusion_acc |   delta |   fixed |   broken |   mean_gate_speech |
|:----------------|:-----------|-------------:|-------------:|--------:|--------:|---------:|-------------------:|
| gated_wavlm_gt  | sd         |        99.9  |        99.95 |    0.05 |       2 |        1 |               0.78 |
| gated_wavlm_gt  | si         |        58.48 |        56.77 |   -1.7  |     742 |      885 |               0.84 |
| gated_wavlm_asr | sd         |        99.9  |       100    |    0.1  |       2 |        0 |               0.8  |
| gated_wavlm_asr | si         |        58.48 |        57.14 |   -1.33 |     813 |      925 |               0.77 |
| concat_wavlm_gt | sd         |        99.9  |       100    |    0.1  |       2 |        0 |             nan    |
| concat_wavlm_gt | si         |        58.48 |        60.12 |    1.64 |    1108 |      970 |             nan    |
| xattn_wavlm_gt  | sd         |        99.9  |        99.9  |    0    |       2 |        2 |             nan    |
| xattn_wavlm_gt  | si         |        58.48 |        63.42 |    4.94 |    1177 |      762 |             nan    |

## SI: gain vs. speech-only confidence (gated_wavlm_asr)

| conf_bin    |    n |   speech_acc |   fusion_acc |   gate |   delta |
|:------------|-----:|-------------:|-------------:|-------:|--------:|
| [0.0, 0.5)  | 1002 |        27.35 |        39.12 |   0.77 |   11.78 |
| [0.5, 0.7)  | 1427 |        32.59 |        37.84 |   0.75 |    5.26 |
| [0.7, 0.9)  | 2497 |        50.1  |        49.58 |   0.76 |   -0.52 |
| [0.9, 0.99) | 3474 |        84.11 |        75.71 |   0.79 |   -8.41 |

## SI: gain vs. speech-only confidence (gated_wavlm_gt)

| conf_bin    |    n |   speech_acc |   fusion_acc |   gate |   delta |
|:------------|-----:|-------------:|-------------:|-------:|--------:|
| [0.0, 0.5)  | 1002 |        27.35 |        32.24 |   0.83 |    4.89 |
| [0.5, 0.7)  | 1427 |        32.59 |        33.71 |   0.84 |    1.12 |
| [0.7, 0.9)  | 2497 |        50.1  |        46.94 |   0.83 |   -3.16 |
| [0.9, 0.99) | 3474 |        84.11 |        80.4  |   0.85 |   -3.71 |
