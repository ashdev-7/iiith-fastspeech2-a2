# Paired significance tests (B vs A on identical splits and seeds)

delta = mean per-run accuracy difference B - A (percentage points) with 95 % t-CI over runs; McNemar pooled over all test clips of all runs (B_fixes = A wrong & B right).

| A                      | B                              | protocol   |   runs |   delta_acc_pp |   ci95_pp |   p_paired_t |   p_wilcoxon |   B_fixes |   B_breaks |   p_mcnemar_pooled |
|:-----------------------|:-------------------------------|:-----------|-------:|---------------:|----------:|-------------:|-------------:|----------:|-----------:|-------------------:|
| speech/logmel          | speech/wavlm                   | rand       |      5 |         0.1905 |    0.4858 |       0.3375 |       0.5    |         7 |          3 |             0.3438 |
| speech/logmel          | speech/wavlm                   | sd         |      5 |         0.381  |    0.6477 |       0.1778 |       0.25   |        10 |          2 |             0.0386 |
| speech/logmel          | speech/wavlm                   | si         |      6 |        16.9048 |   13.2397 |       0.0219 |       0.0312 |      2252 |        832 |             0      |
| speech/wavlm           | fusion/gated_wavlm_gt          | rand       |      5 |         0.1429 |    0.2644 |       0.208  |       0.5    |         3 |          0 |             0.25   |
| speech/wavlm           | fusion/gated_wavlm_gt          | sd         |      5 |         0.0476 |    0.1322 |       0.3739 |       1      |         2 |          1 |             1      |
| speech/wavlm           | fusion/gated_wavlm_gt          | si         |      6 |        -1.7024 |    6.6188 |       0.5378 |       0.6875 |       742 |        885 |             0.0004 |
| speech/wavlm           | fusion/gated_wavlm_asr         | rand       |      5 |         0.0476 |    0.3855 |       0.7489 |       1      |         3 |          2 |             1      |
| speech/wavlm           | fusion/gated_wavlm_asr         | sd         |      5 |         0.0952 |    0.1619 |       0.1778 |       0.5    |         2 |          0 |             0.5    |
| speech/wavlm           | fusion/gated_wavlm_asr         | si         |      6 |        -1.3333 |    9.6351 |       0.7366 |       0.8438 |       813 |        925 |             0.0077 |
| speech/wavlm           | fusion/concat_wavlm_gt         | rand       |      5 |         0      |    0.209  |       1      |       1      |         2 |          2 |             1      |
| speech/wavlm           | fusion/concat_wavlm_gt         | sd         |      5 |         0.0952 |    0.1619 |       0.1778 |       0.5    |         2 |          0 |             0.5    |
| speech/wavlm           | fusion/concat_wavlm_gt         | si         |      6 |         1.6429 |   13.7264 |       0.7707 |       0.8438 |      1108 |        970 |             0.0026 |
| speech/wavlm           | fusion/xattn_wavlm_gt          | rand       |      5 |         0.1429 |    0.2644 |       0.208  |       0.5    |         3 |          0 |             0.25   |
| speech/wavlm           | fusion/xattn_wavlm_gt          | sd         |      5 |         0      |    0.209  |       1      |       1      |         2 |          2 |             1      |
| speech/wavlm           | fusion/xattn_wavlm_gt          | si         |      6 |         4.9405 |   13.6279 |       0.3942 |       0.5625 |      1177 |        762 |             0      |
| speech/wavlm           | speech/wavlm_spknorm           | rand       |      5 |         0.1429 |    0.2644 |       0.208  |       0.5    |         3 |          0 |             0.25   |
| speech/wavlm           | speech/wavlm_spknorm           | sd         |      5 |         0.0476 |    0.2473 |       0.6213 |       1      |         2 |          1 |             1      |
| speech/wavlm           | speech/wavlm_spknorm           | si         |      6 |        -1.9405 |   11.7196 |       0.6881 |       0.5625 |       858 |       1021 |             0.0002 |
| fusion/gated_wavlm_asr | fusion/gated_wavlm_asr_spknorm | rand       |      5 |         0.0476 |    0.2473 |       0.6213 |       1      |         2 |          1 |             1      |
| fusion/gated_wavlm_asr | fusion/gated_wavlm_asr_spknorm | sd         |      5 |        -0.381  |    0.4484 |       0.0777 |       0.125  |         0 |          8 |             0.0078 |
| fusion/gated_wavlm_asr | fusion/gated_wavlm_asr_spknorm | si         |      6 |        -1.7024 |    8.2145 |       0.617  |       0.6875 |       739 |        882 |             0.0004 |
| fusion/gated_wavlm_gt  | fusion/gated_wavlm_asr         | rand       |      5 |        -0.0952 |    0.1619 |       0.1778 |       0.5    |         0 |          2 |             0.5    |
| fusion/gated_wavlm_gt  | fusion/gated_wavlm_asr         | sd         |      5 |         0.0476 |    0.1322 |       0.3739 |       1      |         1 |          0 |             1      |
| fusion/gated_wavlm_gt  | fusion/gated_wavlm_asr         | si         |      6 |         0.369  |    8.1499 |       0.9119 |       0.8438 |       845 |        814 |             0.4614 |

# Calibration (test set, mean over runs)

ECE (15 bins, %), NLL, mean max-probability (%).

| model                                            | protocol   |    ECE |   NLL |   mean_confidence |   runs |
|:-------------------------------------------------|:-----------|-------:|------:|------------------:|-------:|
| Speech: log-Mel + BiGRU (baseline)               | rand       |  3.602 | 0.046 |            96.178 |      5 |
| Speech: log-Mel + BiGRU (baseline)               | sd         |  4.069 | 0.052 |            95.803 |      5 |
| Speech: log-Mel + BiGRU (baseline)               | si         | 38.448 | 2.157 |            79.95  |      6 |
| Speech: WavLM + BiGRU                            | rand       |  3.287 | 0.037 |            96.682 |      5 |
| Speech: WavLM + BiGRU                            | sd         |  3.127 | 0.032 |            96.872 |      5 |
| Speech: WavLM + BiGRU                            | si         | 20.943 | 1.388 |            78.478 |      6 |
| Speech: WavLM + BiGRU + speaker norm (†)         | rand       |  3.353 | 0.034 |            96.647 |      5 |
| Speech: WavLM + BiGRU + speaker norm (†)         | sd         |  3.094 | 0.032 |            96.901 |      5 |
| Speech: WavLM + BiGRU + speaker norm (†)         | si         | 20.084 | 1.387 |            75.944 |      6 |
| Text: DistilRoBERTa + BiGRU (GT transcript)      | rand       |  1.215 | 1.948 |            15.025 |      5 |
| Text: DistilRoBERTa + BiGRU (GT transcript)      | sd         |  0.636 | 1.946 |            14.921 |      5 |
| Text: DistilRoBERTa + BiGRU (GT transcript)      | si         |  0.427 | 1.946 |            14.713 |      6 |
| Text: DistilRoBERTa + BiGRU (Whisper transcript) | rand       |  5.61  | 1.722 |            30.623 |      5 |
| Text: DistilRoBERTa + BiGRU (Whisper transcript) | sd         |  3.238 | 1.697 |            30.05  |      5 |
| Text: DistilRoBERTa + BiGRU (Whisper transcript) | si         | 15.423 | 2.057 |            36.236 |      6 |
| Fusion: concat (GT)                              | rand       |  3.064 | 0.034 |            96.891 |      5 |
| Fusion: concat (GT)                              | sd         |  2.953 | 0.03  |            97.047 |      5 |
| Fusion: concat (GT)                              | si         | 24.29  | 1.447 |            83.569 |      6 |
| Fusion: cross-attention (GT)                     | rand       |  2.903 | 0.03  |            97.097 |      5 |
| Fusion: cross-attention (GT)                     | sd         |  2.884 | 0.032 |            97.075 |      5 |
| Fusion: cross-attention (GT)                     | si         | 21.584 | 1.377 |            82.162 |      6 |
| Fusion: gated GMU (GT)                           | rand       |  3.321 | 0.034 |            96.679 |      5 |
| Fusion: gated GMU (GT)                           | sd         |  3.098 | 0.034 |            96.854 |      5 |
| Fusion: gated GMU (GT)                           | si         | 29.358 | 1.585 |            86.056 |      6 |
| Fusion: gated GMU (Whisper)                      | rand       |  3.325 | 0.035 |            96.694 |      5 |
| Fusion: gated GMU (Whisper)                      | sd         |  3.207 | 0.033 |            96.793 |      5 |
| Fusion: gated GMU (Whisper)                      | si         | 30.231 | 1.676 |            87.223 |      6 |
| Fusion: gated GMU (GT) + speaker norm (†)        | rand       |  3.048 | 0.035 |            96.93  |      5 |
| Fusion: gated GMU (GT) + speaker norm (†)        | sd         |  3.316 | 0.04  |            96.843 |      5 |
| Fusion: gated GMU (GT) + speaker norm (†)        | si         | 26.436 | 1.515 |            82.213 |      6 |
| Fusion: gated GMU (Whisper) + speaker norm (†)   | rand       |  3.189 | 0.035 |            96.774 |      5 |
| Fusion: gated GMU (Whisper) + speaker norm (†)   | sd         |  3.311 | 0.039 |            96.673 |      5 |
| Fusion: gated GMU (Whisper) + speaker norm (†)   | si         | 25.466 | 1.518 |            80.826 |      6 |
