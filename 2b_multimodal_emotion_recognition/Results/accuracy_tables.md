# Accuracy tables

Test-set accuracy and macro-F1 (%), mean ± std over runs. 
**RAND** = clip-level random stratified 70/15/15 split as used in most TESS papers (optimistic: same speakers and words in train and test; 5 seeds). 
**SD** = speaker-dependent, word-disjoint split (5 seeds; 1960/420/420 clips). 
**SI** = speaker-independent leave-one-speaker-out (2 folds × 3 seeds; train 1190 / val 210 / test 1400 clips of the unseen speaker). 
Chance = 14.29 % (7 balanced classes). (†) speaker normalisation uses label-free per-speaker feature statistics, including the unlabelled audio of the test speaker under SI (transductive).

| Model                                                       | RAND Acc      | RAND Macro-F1   | RAND runs   | SD Acc        | SD Macro-F1   | SD runs   | SI Acc        | SI Macro-F1   | SI runs   |
|:------------------------------------------------------------|:--------------|:----------------|:------------|:--------------|:--------------|:----------|:--------------|:--------------|:----------|
| Speech: log-Mel + BiGRU (baseline)                          | 99.67 ± 0.32  | 99.67 ± 0.32    | 5.0         | 99.52 ± 0.56  | 99.52 ± 0.56  | 5.0       | 41.57 ± 5.99  | 36.19 ± 7.62  | 6.0       |
| Speech: WavLM + BiGRU                                       | 99.86 ± 0.21  | 99.86 ± 0.21    | 5.0         | 99.90 ± 0.13  | 99.90 ± 0.13  | 5.0       | 58.48 ± 8.74  | 53.35 ± 8.91  | 6.0       |
| Speech: WavLM + BiGRU + speaker norm (†)                    | 100.00 ± 0.00 | 100.00 ± 0.00   | 5.0         | 99.95 ± 0.11  | 99.95 ± 0.11  | 5.0       | 56.54 ± 5.58  | 51.67 ± 6.63  | 6.0       |
| Text: DistilRoBERTa + BiGRU (GT transcript)                 | 13.81 ± 0.56  | 4.68 ± 0.91     | 5.0         | 14.29 ± 0.00  | 4.67 ± 1.51   | 5.0       | 14.29 ± 0.00  | 8.24 ± 2.05   | 6.0       |
| Text: DistilRoBERTa + BiGRU (Whisper transcript)            | 28.52 ± 1.93  | 21.33 ± 1.36    | 5.0         | 30.05 ± 2.42  | 25.09 ± 3.30  | 5.0       | 23.46 ± 4.72  | 17.42 ± 4.84  | 6.0       |
| Fusion: concat (GT)                                         | 99.86 ± 0.13  | 99.86 ± 0.13    | 5.0         | 100.00 ± 0.00 | 100.00 ± 0.00 | 5.0       | 60.12 ± 11.23 | 55.86 ± 10.38 | 6.0       |
| Fusion: cross-attention (GT)                                | 100.00 ± 0.00 | 100.00 ± 0.00   | 5.0         | 99.90 ± 0.13  | 99.90 ± 0.13  | 5.0       | 63.42 ± 8.84  | 57.01 ± 9.85  | 6.0       |
| Fusion: gated GMU (GT)                                      | 100.00 ± 0.00 | 100.00 ± 0.00   | 5.0         | 99.95 ± 0.11  | 99.95 ± 0.11  | 5.0       | 56.77 ± 7.57  | 52.33 ± 4.69  | 6.0       |
| Fusion: gated GMU (Whisper)                                 | 99.90 ± 0.13  | 99.90 ± 0.13    | 5.0         | 100.00 ± 0.00 | 100.00 ± 0.00 | 5.0       | 57.14 ± 7.29  | 51.43 ± 4.62  | 6.0       |
| Fusion: gated GMU (GT) + speaker norm (†)                   | 99.86 ± 0.21  | 99.86 ± 0.21    | 5.0         | 99.62 ± 0.49  | 99.62 ± 0.49  | 5.0       | 55.82 ± 4.60  | 50.50 ± 4.40  | 6.0       |
| Fusion: gated GMU (Whisper) + speaker norm (†)              | 99.95 ± 0.11  | 99.95 ± 0.11    | 5.0         | 99.62 ± 0.36  | 99.62 ± 0.36  | 5.0       | 55.44 ± 6.28  | 51.22 ± 6.91  | 6.0       |
| Text zero-shot emotion-DistilRoBERTa (GT), no training      | 14.05 ± 0.86  | -               | -           | 14.29 ± 0.00  | -             | -         | 14.29 ± 0.00  | -             | -         |
| Text zero-shot emotion-DistilRoBERTa (Whisper), no training | 14.29 ± 0.48  | -               | -           | 14.19 ± 0.27  | -             | -         | 14.32 ± 0.20  | -             | -         |
| Chance (uniform)                                            | 14.29         | -               | -           | 14.29         | -             | -         | 14.29         | -             | -         |

## Speaker-independent accuracy by direction (%)

| Model                                            | train OAF → test YAF   | train YAF → test OAF   |
|:-------------------------------------------------|:-----------------------|:-----------------------|
| Speech: log-Mel + BiGRU (baseline)               | 43.45 ± 8.68           | 39.69 ± 1.90           |
| Speech: WavLM + BiGRU                            | 53.02 ± 2.51           | 63.93 ± 9.78           |
| Speech: WavLM + BiGRU + speaker norm (†)         | 56.26 ± 7.87           | 56.81 ± 3.95           |
| Text: DistilRoBERTa + BiGRU (GT transcript)      | 14.29 ± 0.00           | 14.29 ± 0.00           |
| Text: DistilRoBERTa + BiGRU (Whisper transcript) | 20.29 ± 4.92           | 26.64 ± 1.13           |
| Fusion: concat (GT)                              | 54.02 ± 14.01          | 66.21 ± 2.72           |
| Fusion: cross-attention (GT)                     | 66.05 ± 11.71          | 60.79 ± 6.13           |
| Fusion: gated GMU (GT)                           | 51.02 ± 6.50           | 62.52 ± 1.43           |
| Fusion: gated GMU (Whisper)                      | 51.40 ± 5.14           | 62.88 ± 2.79           |
| Fusion: gated GMU (GT) + speaker norm (†)        | 54.36 ± 6.58           | 57.29 ± 1.79           |
| Fusion: gated GMU (Whisper) + speaker norm (†)   | 52.29 ± 4.60           | 58.60 ± 6.89           |

## Per-class F1 (%), mean over runs

| Model                                            | Protocol   |   Angry |   Disgust |   Fear |   Happy |   Neutral |   Pleasant surprise |   Sad |
|:-------------------------------------------------|:-----------|--------:|----------:|-------:|--------:|----------:|--------------------:|------:|
| Speech: log-Mel + BiGRU (baseline)               | RAND       |   100   |      99.8 |  100   |    99.3 |      99.7 |                99.3 |  99.5 |
| Speech: log-Mel + BiGRU (baseline)               | SD         |    99.7 |      99.3 |   99.8 |    98.8 |      99.8 |                99.5 |  99.7 |
| Speech: log-Mel + BiGRU (baseline)               | SI         |    20.7 |      51   |   30.9 |     0.6 |      83.8 |                13.6 |  52.7 |
| Speech: WavLM + BiGRU                            | RAND       |    99.8 |      99.8 |  100   |    99.7 |     100   |                99.7 | 100   |
| Speech: WavLM + BiGRU                            | SD         |   100   |      99.7 |  100   |   100   |     100   |                99.7 | 100   |
| Speech: WavLM + BiGRU                            | SI         |    15   |      74.9 |   66.1 |     6.6 |      89.9 |                36.6 |  84.4 |
| Speech: WavLM + BiGRU + speaker norm (†)         | RAND       |   100   |     100   |  100   |   100   |     100   |               100   | 100   |
| Speech: WavLM + BiGRU + speaker norm (†)         | SD         |   100   |      99.8 |  100   |   100   |     100   |                99.8 | 100   |
| Speech: WavLM + BiGRU + speaker norm (†)         | SI         |    20.1 |      74.9 |   74   |     2.7 |      84.2 |                36.5 |  69.3 |
| Text: DistilRoBERTa + BiGRU (GT transcript)      | RAND       |    10.1 |      10.8 |    5   |     0   |       1.6 |                 1.1 |   4.2 |
| Text: DistilRoBERTa + BiGRU (GT transcript)      | SD         |     0   |       0   |    8.6 |     0   |      14.6 |                 0   |   9.5 |
| Text: DistilRoBERTa + BiGRU (GT transcript)      | SI         |     7.8 |       0   |    7   |    13.8 |      11.4 |                 7   |  10.7 |
| Text: DistilRoBERTa + BiGRU (Whisper transcript) | RAND       |    11.7 |      21.2 |   21.4 |     0.6 |      33.2 |                58.3 |   2.8 |
| Text: DistilRoBERTa + BiGRU (Whisper transcript) | SD         |    28.9 |      13.8 |   26   |     0   |      38.6 |                55.7 |  12.6 |
| Text: DistilRoBERTa + BiGRU (Whisper transcript) | SI         |    16.7 |      12.1 |   18.6 |     1.1 |      25.7 |                35.6 |  12.2 |
| Fusion: concat (GT)                              | RAND       |    99.8 |     100   |  100   |    99.7 |     100   |                99.7 |  99.8 |
| Fusion: concat (GT)                              | SD         |   100   |     100   |  100   |   100   |     100   |               100   | 100   |
| Fusion: concat (GT)                              | SI         |    39.6 |      68.4 |   78.7 |     1.6 |      94.2 |                35.7 |  72.9 |
| Fusion: cross-attention (GT)                     | RAND       |   100   |     100   |  100   |   100   |     100   |               100   | 100   |
| Fusion: cross-attention (GT)                     | SD         |   100   |      99.8 |  100   |    99.8 |     100   |                99.7 | 100   |
| Fusion: cross-attention (GT)                     | SI         |    24.6 |      69.6 |   77.1 |     6.3 |      95.1 |                43.9 |  82.5 |
| Fusion: gated GMU (GT)                           | RAND       |   100   |     100   |  100   |   100   |     100   |               100   | 100   |
| Fusion: gated GMU (GT)                           | SD         |   100   |     100   |  100   |    99.8 |     100   |                99.8 | 100   |
| Fusion: gated GMU (GT)                           | SI         |    23.1 |      59.2 |   70.1 |     6.9 |      90.7 |                24.6 |  91.8 |
| Fusion: gated GMU (Whisper)                      | RAND       |    99.8 |     100   |   99.8 |    99.8 |     100   |                99.8 | 100   |
| Fusion: gated GMU (Whisper)                      | SD         |   100   |     100   |  100   |   100   |     100   |               100   | 100   |
| Fusion: gated GMU (Whisper)                      | SI         |    19.4 |      80.2 |   70.1 |     4.8 |      87.5 |                27.8 |  70.2 |
| Fusion: gated GMU (GT) + speaker norm (†)        | RAND       |   100   |      99.8 |  100   |    99.7 |     100   |                99.5 | 100   |
| Fusion: gated GMU (GT) + speaker norm (†)        | SD         |   100   |      99.5 |  100   |    99.3 |      99.8 |                98.9 |  99.8 |
| Fusion: gated GMU (GT) + speaker norm (†)        | SI         |    26.3 |      76.3 |   61.3 |     9.7 |      89.4 |                21.6 |  68.9 |
| Fusion: gated GMU (Whisper) + speaker norm (†)   | RAND       |    99.8 |     100   |   99.8 |   100   |     100   |               100   | 100   |
| Fusion: gated GMU (Whisper) + speaker norm (†)   | SD         |   100   |      99.2 |  100   |    99.5 |     100   |                99.2 |  99.5 |
| Fusion: gated GMU (Whisper) + speaker norm (†)   | SI         |    15   |      70   |   70.8 |    15.3 |      87.4 |                18.1 |  81.9 |
