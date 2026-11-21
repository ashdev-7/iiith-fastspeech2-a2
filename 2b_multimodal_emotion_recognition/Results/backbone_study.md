# Pre-registered encoder study (SI)

Plan: `analysis/PREREGISTRATION_si_backbones.md` (pushed before the run). † transductive per-speaker normalisation; I = inductive (test-speaker statistics from 40 calibration words only, scored on the other 160 words).

## Audits

* D1 (word identity carries no label information): PASS: every (speaker, word) pair has each of the 7 emotions exactly once (400 pairs).
* D2 (OAF->YAF): PASS: training rows only OAF, test rows only YAF, no shared clip.
* D2 (YAF->OAF): PASS: training rows only YAF, test rows only OAF, no shared clip.
* D3: PASS by construction: `zscore()` receives features and row indices only.
* D4: PASS by construction: perturbed copies are indexed with training rows only.
* C1 (A-L1n(B) reproduces study 3): PASS: 100.00 % identical predictions (2800 clips).

## Accuracy (%)

| model                                   | direction   |    acc |   acc_160 |   acc_inductive_160 |   mean_of_directions |
|:----------------------------------------|:------------|-------:|----------:|--------------------:|---------------------:|
| A-L1n(B)                                | OAF->YAF    |  76.36 |     76.61 |               76.52 |                72.04 |
| L1n(B)                                  | OAF->YAF    |  75.29 |     75.62 |               75.27 |                70.65 |
| A-L1n(WL)                               | OAF->YAF    |  80    |     80.09 |               80.09 |                75.68 |
| L1n(WL)                                 | OAF->YAF    |  79.79 |     79.82 |               79.91 |                75.04 |
| A-L1n(HL)                               | OAF->YAF    |  75.79 |     76.25 |               75.98 |                71.22 |
| L1n(HL)                                 | OAF->YAF    |  74.86 |     75.54 |               75.18 |                70.82 |
| A-L1n(E)                                | OAF->YAF    |  68.29 |     68.21 |               67.68 |                68.04 |
| L1n(E)                                  | OAF->YAF    |  66.64 |     66.96 |               66.25 |                65.6  |
| A-L1n(LC)                               | OAF->YAF    | 100    |    100    |              100    |                99.9  |
| L1n(LC)                                 | OAF->YAF    |  99.79 |     99.73 |               99.73 |                99.08 |
| ENS                                     | OAF->YAF    |  77.79 |     78.04 |               77.95 |                72.93 |
| LC-a zero-shot (emotion2vec+ seed head) | OAF->YAF    |  85.71 |    nan    |              nan    |                85.71 |
| A-L1n(B)                                | YAF->OAF    |  67.71 |     67.86 |               67.5  |                72.04 |
| L1n(B)                                  | YAF->OAF    |  66    |     65.8  |               65.18 |                70.65 |
| A-L1n(WL)                               | YAF->OAF    |  71.36 |     71.25 |               71.16 |                75.68 |
| L1n(WL)                                 | YAF->OAF    |  70.29 |     70.18 |               70    |                75.04 |
| A-L1n(HL)                               | YAF->OAF    |  66.64 |     66.52 |               66.43 |                71.22 |
| L1n(HL)                                 | YAF->OAF    |  66.79 |     66.88 |               66.79 |                70.82 |
| A-L1n(E)                                | YAF->OAF    |  67.79 |     67.68 |               68.93 |                68.04 |
| L1n(E)                                  | YAF->OAF    |  64.57 |     64.46 |               66.52 |                65.6  |
| A-L1n(LC)                               | YAF->OAF    |  99.79 |     99.73 |               99.82 |                99.9  |
| L1n(LC)                                 | YAF->OAF    |  98.36 |     98.57 |               98.57 |                99.08 |
| ENS                                     | YAF->OAF    |  68.07 |     67.95 |               67.95 |                72.93 |
| LC-a zero-shot (emotion2vec+ seed head) | YAF->OAF    |  85.71 |    nan    |              nan    |                85.71 |

## Pre-registered tests

Word-cluster bootstrap (10 000 resamples, seed 0); Holm over HB1 + HB2 (8 tests); verdicts with a ROPE of +-2 pp.

| hypothesis   | comparison                                   | direction   |   delta_pp |   ci_lo |   ci_hi |   p_boot |   p_holm | verdict             |
|:-------------|:---------------------------------------------|:------------|-----------:|--------:|--------:|---------:|---------:|:--------------------|
| HB1          | ENS - A-L1n(B)                               | OAF->YAF    |       1.43 |    0.43 |    2.43 |   0.0052 |   0.026  | direction-specific  |
| HB1          | ENS - A-L1n(B)                               | YAF->OAF    |       0.36 |   -0.43 |    1.14 |   0.434  |   1      | direction-specific  |
| HB2          | A-L1n(WL) - A-L1n(B)                         | OAF->YAF    |       3.64 |    2.64 |    4.71 |   0      |   0      | replicated (+)      |
| HB2          | A-L1n(WL) - A-L1n(B)                         | YAF->OAF    |       3.64 |    2.64 |    4.64 |   0      |   0      | replicated (+)      |
| HB2          | A-L1n(HL) - A-L1n(B)                         | OAF->YAF    |      -0.57 |   -1.79 |    0.64 |   0.377  |   1      | equivalent (+-2 pp) |
| HB2          | A-L1n(HL) - A-L1n(B)                         | YAF->OAF    |      -1.07 |   -2    |   -0.14 |   0.0314 |   0.1256 | equivalent (+-2 pp) |
| HB2          | A-L1n(E) - A-L1n(B)                          | OAF->YAF    |      -8.07 |  -10.07 |   -6.07 |   0      |   0      | direction-specific  |
| HB2          | A-L1n(E) - A-L1n(B)                          | YAF->OAF    |       0.07 |   -1.79 |    1.86 |   0.9582 |   1      | direction-specific  |
| HB3          | ENS inductive - ENS transductive (160 words) | OAF->YAF    |      -0.09 |   -0.45 |    0.18 |   0.7756 | nan      | equivalent (+-2 pp) |
| HB3          | ENS inductive - ENS transductive (160 words) | YAF->OAF    |       0    |   -0.27 |    0.27 |   1      | nan      | equivalent (+-2 pp) |

* **HB1 (primary, ENS > A-L1n(B) in both directions): NOT SUPPORTED** (direction-specific).
* **HB3 (inductive ENS within 2 pp of transductive): SUPPORTED.**
* LC (emotion2vec+ seed) is a leakage control, never a candidate; its rows only show what label exposure that may include TESS produces.

## POST HOC: accuracy against C (diagnostic only; reads the test speaker; never used to choose)

| model     |      C |   OAF->YAF |   YAF->OAF |   mean |
|:----------|-------:|-----------:|-----------:|-------:|
| A-L1n(B)  | 0.0001 |      74.36 |      66.14 |  70.25 |
| A-L1n(B)  | 0.0003 |      75.64 |      66.43 |  71.04 |
| A-L1n(B)  | 0.001  |      76.36 |      67.71 |  72.04 |
| A-L1n(B)  | 0.003  |      76.86 |      68.64 |  72.75 |
| A-L1n(B)  | 0.01   |      77.07 |      69    |  73.04 |
| A-L1n(B)  | 0.03   |      77.29 |      67.57 |  72.43 |
| A-L1n(B)  | 0.1    |      74.64 |      68.43 |  71.54 |
| A-L1n(WL) | 0.0001 |      79    |      69.93 |  74.46 |
| A-L1n(WL) | 0.0003 |      79.79 |      70.71 |  75.25 |
| A-L1n(WL) | 0.001  |      80    |      71.36 |  75.68 |
| A-L1n(WL) | 0.003  |      80.29 |      71.71 |  76    |
| A-L1n(WL) | 0.01   |      79.79 |      72.43 |  76.11 |
| A-L1n(WL) | 0.03   |      79.14 |      71    |  75.07 |
| A-L1n(WL) | 0.1    |      78.86 |      70.64 |  74.75 |
| A-L1n(HL) | 0.0001 |      75.36 |      66.07 |  70.71 |
| A-L1n(HL) | 0.0003 |      75.29 |      66.43 |  70.86 |
| A-L1n(HL) | 0.001  |      75.79 |      66.64 |  71.21 |
| A-L1n(HL) | 0.003  |      75.64 |      66.79 |  71.21 |
| A-L1n(HL) | 0.01   |      76.86 |      70.36 |  73.61 |
| A-L1n(HL) | 0.03   |      76.14 |      60.29 |  68.21 |
| A-L1n(HL) | 0.1    |      71.64 |      60.21 |  65.93 |
| A-L1n(E)  | 0.0001 |      65.93 |      62.21 |  64.07 |
| A-L1n(E)  | 0.0003 |      67.79 |      65.21 |  66.5  |
| A-L1n(E)  | 0.001  |      68.29 |      67.79 |  68.04 |
| A-L1n(E)  | 0.003  |      68.64 |      69.57 |  69.11 |
| A-L1n(E)  | 0.01   |      69.07 |      70.93 |  70    |
| A-L1n(E)  | 0.03   |      69.86 |      71.29 |  70.57 |
| A-L1n(E)  | 0.1    |      70.29 |      72.43 |  71.36 |
| A-L1n(LC) | 0.0001 |     100    |      99.79 |  99.89 |
| A-L1n(LC) | 0.0003 |     100    |      99.79 |  99.89 |
| A-L1n(LC) | 0.001  |     100    |      99.79 |  99.89 |
| A-L1n(LC) | 0.003  |     100    |      99.86 |  99.93 |
| A-L1n(LC) | 0.01   |      99.93 |      99.86 |  99.89 |
| A-L1n(LC) | 0.03   |      99.86 |      99.86 |  99.86 |
| A-L1n(LC) | 0.1    |      99.93 |      99.86 |  99.89 |

