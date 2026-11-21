# POST-HOC sensitivity of the pre-registered linear model to C

Not pre-registered: every row reads the test speaker. The pre-registered value is C = 0.001 (chosen by tie-break; training-speaker CV was 100 % for every C). Diagnostic only.

| model                |       C |   OAF->YAF |   YAF->OAF |   mean |
|:---------------------|--------:|-----------:|-----------:|-------:|
| L1 (no norm)         |  1e-05  |      63.64 |      63.36 |  63.5  |
| L1 (no norm)         |  3e-05  |      65.86 |      63.29 |  64.58 |
| L1 (no norm)         |  0.0001 |      66.86 |      63.57 |  65.22 |
| L1 (no norm)         |  0.0003 |      67.79 |      63.29 |  65.54 |
| L1 (no norm)         |  0.001  |      68.43 |      63.5  |  65.96 |
| L1 (no norm)         |  0.003  |      69.07 |      64.07 |  66.57 |
| L1 (no norm)         |  0.01   |      69.07 |      65.14 |  67.1  |
| L1 (no norm)         |  0.03   |      68.79 |      65.43 |  67.11 |
| L1 (no norm)         |  0.1    |      69.64 |      62.64 |  66.14 |
| L1 (no norm)         |  0.3    |      69.43 |      66.79 |  68.11 |
| L1 (no norm)         |  1      |      69.43 |      66.79 |  68.11 |
| L1 (no norm)         | 10      |      69.43 |      66.79 |  68.11 |
| L1n (speaker norm †) |  1e-05  |      69.21 |      64.21 |  66.71 |
| L1n (speaker norm †) |  3e-05  |      72.07 |      64.21 |  68.14 |
| L1n (speaker norm †) |  0.0001 |      73.79 |      64.64 |  69.22 |
| L1n (speaker norm †) |  0.0003 |      74.5  |      65.21 |  69.85 |
| L1n (speaker norm †) |  0.001  |      75.29 |      66    |  70.65 |
| L1n (speaker norm †) |  0.003  |      76.07 |      66.5  |  71.28 |
| L1n (speaker norm †) |  0.01   |      76.86 |      67.93 |  72.4  |
| L1n (speaker norm †) |  0.03   |      76.5  |      68.86 |  72.68 |
| L1n (speaker norm †) |  0.1    |      76.07 |      67.36 |  71.72 |
| L1n (speaker norm †) |  0.3    |      75.21 |      67.86 |  71.54 |
| L1n (speaker norm †) |  1      |      75.29 |      67.86 |  71.58 |
| L1n (speaker norm †) | 10      |      75.29 |      67.86 |  71.58 |
