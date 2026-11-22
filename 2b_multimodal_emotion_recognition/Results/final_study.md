# Pre-registered final study (SI)

Plan: `analysis/PREREGISTRATION_si_final.md` (pushed before the run). † transductive (test-speaker statistics and self-training pool from all its unlabelled clips); I = inductive (both from the 40 calibration words only, scored on the other 160 words).

## Audits

* D1 (word identity carries no label information): PASS: every (speaker, word) pair has each of the 7 emotions exactly once (400 pairs).
* D2 (OAF->YAF): PASS: training rows only OAF, test rows only YAF, no shared clip.
* D2 (YAF->OAF): PASS: training rows only YAF, test rows only OAF, no shared clip.
* D3: PASS by construction: `zscore()` receives features and row indices only.
* D4: PASS by construction: perturbed copies are indexed with training rows only.
* D5 (inductive: only calibration-word clips are ever pseudo-labelled): PASS.
* D6: PASS by construction: `run_direction()` reads the training speaker's labels only; test labels are read in `main()` after all predictions are fixed.
* C1 (A-L1n(WL) reproduces study 4, >= 99 % identical): PASS: 100.00 % identical predictions (2800 clips).

## Accuracy (%)

| model         | direction   |   acc |   acc_160 |   acc_inductive_160 |   mean_of_directions |   mean_inductive_160 |
|:--------------|:------------|------:|----------:|--------------------:|---------------------:|---------------------:|
| A-L1n(WL)     | OAF->YAF    | 80    |     80.09 |               80.09 |                75.68 |                75.62 |
| AP-L1n(WL)    | OAF->YAF    | 77.29 |     77.32 |               77.14 |                74.46 |                74.24 |
| A-L1n(W3)     | OAF->YAF    | 78    |     77.95 |               77.5  |                75.93 |                75.58 |
| AP-L1n(W3)    | OAF->YAF    | 76.93 |     76.7  |               76.16 |                75.68 |                75.13 |
| ENS2          | OAF->YAF    | 77.79 |     77.77 |               77.23 |                75.68 |                75.26 |
| ST(A-L1n(WL)) | OAF->YAF    | 82.36 |     82.41 |               79.29 |                72.68 |                74.42 |
| FINAL         | OAF->YAF    | 75.64 |     75.89 |               75.89 |                73.6  |                73.75 |
| A-L1n(WL)     | YAF->OAF    | 71.36 |     71.25 |               71.16 |                75.68 |                75.62 |
| AP-L1n(WL)    | YAF->OAF    | 71.64 |     71.52 |               71.34 |                74.46 |                74.24 |
| A-L1n(W3)     | YAF->OAF    | 73.86 |     73.84 |               73.66 |                75.93 |                75.58 |
| AP-L1n(W3)    | YAF->OAF    | 74.43 |     74.55 |               74.11 |                75.68 |                75.13 |
| ENS2          | YAF->OAF    | 73.57 |     73.57 |               73.3  |                75.68 |                75.26 |
| ST(A-L1n(WL)) | YAF->OAF    | 63    |     63.04 |               69.55 |                72.68 |                74.42 |
| FINAL         | YAF->OAF    | 71.57 |     71.52 |               71.61 |                73.6  |                73.75 |

## Pre-registered tests

Word-cluster bootstrap (10 000 resamples, seed 0); Holm over H1 + H2 (8 tests); verdicts with a ROPE of +-2 pp.

| hypothesis   | comparison                                       | direction   |   delta_pp |   ci_lo |   ci_hi |   p_boot |   p_holm | verdict                       |
|:-------------|:-------------------------------------------------|:------------|-----------:|--------:|--------:|---------:|---------:|:------------------------------|
| H1           | FINAL - A-L1n(WL)                                | OAF->YAF    |      -4.36 |   -5.29 |   -3.5  |   0      |        0 | direction-specific            |
| H1           | FINAL - A-L1n(WL)                                | YAF->OAF    |       0.21 |   -0.71 |    1.14 |   0.6736 |        1 | direction-specific            |
| H2-W         | A-L1n(W3) - A-L1n(WL)                            | OAF->YAF    |      -2    |   -2.86 |   -1.21 |   0      |        0 | inconclusive (opposite signs) |
| H2-W         | A-L1n(W3) - A-L1n(WL)                            | YAF->OAF    |       2.5  |    1.43 |    3.57 |   0      |        0 | inconclusive (opposite signs) |
| H2-P         | AP-L1n(WL) - A-L1n(WL)                           | OAF->YAF    |      -2.71 |   -3.5  |   -2    |   0      |        0 | direction-specific            |
| H2-P         | AP-L1n(WL) - A-L1n(WL)                           | YAF->OAF    |       0.29 |   -0.64 |    1.21 |   0.573  |        1 | direction-specific            |
| H2-S         | ST(A-L1n(WL)) - A-L1n(WL)                        | OAF->YAF    |       2.36 |    1.64 |    3.14 |   0      |        0 | inconclusive (opposite signs) |
| H2-S         | ST(A-L1n(WL)) - A-L1n(WL)                        | YAF->OAF    |      -8.36 |   -9.57 |   -7.14 |   0      |        0 | inconclusive (opposite signs) |
| H3           | FINAL inductive - FINAL transductive (160 words) | OAF->YAF    |       0    |   -0.54 |    0.54 |   1      |      nan | equivalent (+-2 pp)           |
| H3           | FINAL inductive - FINAL transductive (160 words) | YAF->OAF    |       0.09 |   -1.16 |    1.25 |   0.9188 |      nan | equivalent (+-2 pp)           |

* **H1 (primary, FINAL > A-L1n(WL) in both directions): NOT SUPPORTED** (direction-specific).
* **H3 (inductive FINAL within 2 pp of transductive): SUPPORTED.**
* **Best system by the registered reporting rule: A-L1n(WL).** No other model is promoted, whatever it scores.

## POST HOC diagnostic: accuracy of the pseudo-labels per self-training round (reads test labels after every prediction is fixed; never used to choose)

| model         | mode   | direction   |   round |   n_pseudo |   pseudo_label_acc |
|:--------------|:-------|:------------|--------:|-----------:|-------------------:|
| ST(A-L1n(WL)) | T      | OAF->YAF    |       1 |        350 |              99.71 |
| ST(A-L1n(WL)) | T      | OAF->YAF    |       2 |        700 |              99.86 |
| ST(A-L1n(WL)) | T      | OAF->YAF    |       3 |       1050 |              94.19 |
| ST(A-L1n(WL)) | I      | OAF->YAF    |       1 |         70 |             100    |
| ST(A-L1n(WL)) | I      | OAF->YAF    |       2 |        140 |             100    |
| ST(A-L1n(WL)) | I      | OAF->YAF    |       3 |        210 |              94.29 |
| FINAL         | T      | OAF->YAF    |       1 |        350 |              98.29 |
| FINAL         | T      | OAF->YAF    |       2 |        700 |              93.86 |
| FINAL         | T      | OAF->YAF    |       3 |       1050 |              86.86 |
| FINAL         | I      | OAF->YAF    |       1 |         70 |              98.57 |
| FINAL         | I      | OAF->YAF    |       2 |        140 |              92.86 |
| FINAL         | I      | OAF->YAF    |       3 |        210 |              87.62 |
| ST(A-L1n(WL)) | T      | YAF->OAF    |       1 |        350 |              98.57 |
| ST(A-L1n(WL)) | T      | YAF->OAF    |       2 |        700 |              99.14 |
| ST(A-L1n(WL)) | T      | YAF->OAF    |       3 |       1050 |              79.24 |
| ST(A-L1n(WL)) | I      | YAF->OAF    |       1 |         70 |             100    |
| ST(A-L1n(WL)) | I      | YAF->OAF    |       2 |        140 |              99.29 |
| ST(A-L1n(WL)) | I      | YAF->OAF    |       3 |        210 |              80    |
| FINAL         | T      | YAF->OAF    |       1 |        350 |              99.71 |
| FINAL         | T      | YAF->OAF    |       2 |        700 |              99.86 |
| FINAL         | T      | YAF->OAF    |       3 |       1050 |              82.19 |
| FINAL         | I      | YAF->OAF    |       1 |         70 |             100    |
| FINAL         | I      | YAF->OAF    |       2 |        140 |              99.29 |
| FINAL         | I      | YAF->OAF    |       3 |        210 |              82.38 |

