# Regeneration of the noise study: checks N1 and N2

Plan: `analysis/PREREGISTRATION_si_augmentation.md`, part N.

## N1a: Whisper transcripts

2800 of 2800 clean transcripts identical to `artifacts/asr_transcripts.csv` (100.00 %).

## N1b: retrained vs. committed runs (66 runs)

Runs with 100 % identical predictions: 66 of 66.

| model                          | run            |   acc_committed |   acc_retrained |   identical_pred_pct |
|:-------------------------------|:---------------|----------------:|----------------:|---------------------:|
| speech/wavlm                   | sd_fold0_seed0 |           99.76 |           99.76 |                  100 |
| speech/wavlm                   | sd_fold0_seed1 |          100    |          100    |                  100 |
| speech/wavlm                   | sd_fold0_seed2 |           99.76 |           99.76 |                  100 |
| speech/wavlm                   | sd_fold0_seed3 |          100    |          100    |                  100 |
| speech/wavlm                   | sd_fold0_seed4 |          100    |          100    |                  100 |
| speech/wavlm                   | si_fold0_seed0 |           52.14 |           52.14 |                  100 |
| speech/wavlm                   | si_fold0_seed1 |           51.07 |           51.07 |                  100 |
| speech/wavlm                   | si_fold0_seed2 |           55.86 |           55.86 |                  100 |
| speech/wavlm                   | si_fold1_seed0 |           61.43 |           61.43 |                  100 |
| speech/wavlm                   | si_fold1_seed1 |           74.71 |           74.71 |                  100 |
| speech/wavlm                   | si_fold1_seed2 |           55.64 |           55.64 |                  100 |
| speech/wavlm_spknorm           | sd_fold0_seed0 |          100    |          100    |                  100 |
| speech/wavlm_spknorm           | sd_fold0_seed1 |          100    |          100    |                  100 |
| speech/wavlm_spknorm           | sd_fold0_seed2 |          100    |          100    |                  100 |
| speech/wavlm_spknorm           | sd_fold0_seed3 |          100    |          100    |                  100 |
| speech/wavlm_spknorm           | sd_fold0_seed4 |           99.76 |           99.76 |                  100 |
| speech/wavlm_spknorm           | si_fold0_seed0 |           52.71 |           52.71 |                  100 |
| speech/wavlm_spknorm           | si_fold0_seed1 |           65.29 |           65.29 |                  100 |
| speech/wavlm_spknorm           | si_fold0_seed2 |           50.79 |           50.79 |                  100 |
| speech/wavlm_spknorm           | si_fold1_seed0 |           61.36 |           61.36 |                  100 |
| speech/wavlm_spknorm           | si_fold1_seed1 |           54.29 |           54.29 |                  100 |
| speech/wavlm_spknorm           | si_fold1_seed2 |           54.79 |           54.79 |                  100 |
| text/asr                       | sd_fold0_seed0 |           30.48 |           30.48 |                  100 |
| text/asr                       | sd_fold0_seed1 |           30.71 |           30.71 |                  100 |
| text/asr                       | sd_fold0_seed2 |           26.19 |           26.19 |                  100 |
| text/asr                       | sd_fold0_seed3 |           30    |           30    |                  100 |
| text/asr                       | sd_fold0_seed4 |           32.86 |           32.86 |                  100 |
| text/asr                       | si_fold0_seed0 |           18    |           18    |                  100 |
| text/asr                       | si_fold0_seed1 |           25.93 |           25.93 |                  100 |
| text/asr                       | si_fold0_seed2 |           16.93 |           16.93 |                  100 |
| text/asr                       | si_fold1_seed0 |           25.36 |           25.36 |                  100 |
| text/asr                       | si_fold1_seed1 |           27.07 |           27.07 |                  100 |
| text/asr                       | si_fold1_seed2 |           27.5  |           27.5  |                  100 |
| fusion/gated_wavlm_gt          | sd_fold0_seed0 |           99.76 |           99.76 |                  100 |
| fusion/gated_wavlm_gt          | sd_fold0_seed1 |          100    |          100    |                  100 |
| fusion/gated_wavlm_gt          | sd_fold0_seed2 |          100    |          100    |                  100 |
| fusion/gated_wavlm_gt          | sd_fold0_seed3 |          100    |          100    |                  100 |
| fusion/gated_wavlm_gt          | sd_fold0_seed4 |          100    |          100    |                  100 |
| fusion/gated_wavlm_gt          | si_fold0_seed0 |           53.21 |           53.21 |                  100 |
| fusion/gated_wavlm_gt          | si_fold0_seed1 |           43.71 |           43.71 |                  100 |
| fusion/gated_wavlm_gt          | si_fold0_seed2 |           56.14 |           56.14 |                  100 |
| fusion/gated_wavlm_gt          | si_fold1_seed0 |           61.14 |           61.14 |                  100 |
| fusion/gated_wavlm_gt          | si_fold1_seed1 |           64    |           64    |                  100 |
| fusion/gated_wavlm_gt          | si_fold1_seed2 |           62.43 |           62.43 |                  100 |
| fusion/gated_wavlm_asr         | sd_fold0_seed0 |          100    |          100    |                  100 |
| fusion/gated_wavlm_asr         | sd_fold0_seed1 |          100    |          100    |                  100 |
| fusion/gated_wavlm_asr         | sd_fold0_seed2 |          100    |          100    |                  100 |
| fusion/gated_wavlm_asr         | sd_fold0_seed3 |          100    |          100    |                  100 |
| fusion/gated_wavlm_asr         | sd_fold0_seed4 |          100    |          100    |                  100 |
| fusion/gated_wavlm_asr         | si_fold0_seed0 |           49.14 |           49.14 |                  100 |
| fusion/gated_wavlm_asr         | si_fold0_seed1 |           57.29 |           57.29 |                  100 |
| fusion/gated_wavlm_asr         | si_fold0_seed2 |           47.79 |           47.79 |                  100 |
| fusion/gated_wavlm_asr         | si_fold1_seed0 |           63.64 |           63.64 |                  100 |
| fusion/gated_wavlm_asr         | si_fold1_seed1 |           59.79 |           59.79 |                  100 |
| fusion/gated_wavlm_asr         | si_fold1_seed2 |           65.21 |           65.21 |                  100 |
| fusion/gated_wavlm_asr_spknorm | sd_fold0_seed0 |           99.76 |           99.76 |                  100 |
| fusion/gated_wavlm_asr_spknorm | sd_fold0_seed1 |           99.76 |           99.76 |                  100 |
| fusion/gated_wavlm_asr_spknorm | sd_fold0_seed2 |          100    |          100    |                  100 |
| fusion/gated_wavlm_asr_spknorm | sd_fold0_seed3 |           99.52 |           99.52 |                  100 |
| fusion/gated_wavlm_asr_spknorm | sd_fold0_seed4 |           99.05 |           99.05 |                  100 |
| fusion/gated_wavlm_asr_spknorm | si_fold0_seed0 |           55.79 |           55.79 |                  100 |
| fusion/gated_wavlm_asr_spknorm | si_fold0_seed1 |           47.07 |           47.07 |                  100 |
| fusion/gated_wavlm_asr_spknorm | si_fold0_seed2 |           54    |           54    |                  100 |
| fusion/gated_wavlm_asr_spknorm | si_fold1_seed0 |           52.14 |           52.14 |                  100 |
| fusion/gated_wavlm_asr_spknorm | si_fold1_seed1 |           57.79 |           57.79 |                  100 |
| fusion/gated_wavlm_asr_spknorm | si_fold1_seed2 |           65.86 |           65.86 |                  100 |

## N2: regenerated vs. log-recovered per-run noise accuracies

330 cells; 330 matched; max |difference| 0.05 pp; 0 cells differ by more than 0.1 pp or are unmatched.


