# Pre-registered SI experiment: all-layer pooled WavLM + logistic regression

Plan: `analysis/PREREGISTRATION_si_linear.md`. C chosen by word-grouped 5-fold CV on the training speaker only. † = transductive (label-free) speaker normalisation.

## Per direction

| model                             | train   | test   |     C |   cv_acc_train_speaker |   accuracy |   macro_f1 |
|:----------------------------------|:--------|:-------|------:|-----------------------:|-----------:|-----------:|
| all-layer linear                  | OAF     | YAF    | 0.001 |                    100 |      68.43 |      66.31 |
| all-layer linear                  | YAF     | OAF    | 0.001 |                    100 |      63.5  |      55.59 |
| all-layer linear + speaker norm † | OAF     | YAF    | 0.001 |                    100 |      75.29 |      71.27 |
| all-layer linear + speaker norm † | YAF     | OAF    | 0.001 |                    100 |      66    |      61.93 |

## Mean over the two directions

| model                             |   accuracy |   macro_f1 |
|:----------------------------------|-----------:|-----------:|
| all-layer linear                  |      65.96 |      60.95 |
| all-layer linear + speaker norm † |      70.65 |      66.6  |

## Against the 6 SI runs of the BiGRU models (same test clips)

|                                                                         |   mean_delta_pp | runs_better   |   max_p_mcnemar |
|:------------------------------------------------------------------------|----------------:|:--------------|----------------:|
| ('all-layer linear', 'WavLM + BiGRU')                                   |          7.49   | 5/6           |          0.0106 |
| ('all-layer linear + speaker norm †', 'WavLM + BiGRU')                  |         12.1667 | 5/6           |          0      |
| ('all-layer linear', 'WavLM + BiGRU + speaker norm †')                  |          9.425  | 6/6           |          0.0461 |
| ('all-layer linear + speaker norm †', 'WavLM + BiGRU + speaker norm †') |         14.105  | 6/6           |          0      |

Per-run McNemar results: `si_linear_vs_bigru.csv`.
