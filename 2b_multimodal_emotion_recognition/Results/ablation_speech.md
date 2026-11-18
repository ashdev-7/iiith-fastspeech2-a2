# Speech front-end ablation (test accuracy %, paired against the main WavLM model)

delta = mean per-run difference vs `wavlm` on identical splits/seeds, with 95 % t-CI and paired t-test. max_layer_weight = mean over runs of the largest softmax layer weight (uniform = 1/L).

| variant      | description                    | protocol   |   runs |   acc_mean |   acc_std |   max_layer_weight |   uniform_weight |   top_layer |   paired_runs |   delta_pp |   ci95_pp |   p_paired_t |
|:-------------|:-------------------------------|:-----------|-------:|-----------:|----------:|-------------------:|-----------------:|------------:|--------------:|-----------:|----------:|-------------:|
| wavlm        | main: 7 layers, 40 ms          | sd         |      5 |     99.905 |     0.13  |              0.157 |            0.143 |           2 |           nan |    nan     |   nan     |      nan     |
| wavlm        | main: 7 layers, 40 ms          | si         |      6 |     58.476 |     8.743 |              0.156 |            0.143 |           0 |           nan |    nan     |   nan     |      nan     |
| wavlm        | main: 7 layers, 40 ms          | rand       |      5 |     99.857 |     0.213 |              0.16  |            0.143 |           0 |           nan |    nan     |   nan     |      nan     |
| wavlm_full   | all 13 layers, 20 ms           | sd         |      5 |     99.857 |     0.213 |              0.088 |            0.077 |           1 |             5 |     -0.048 |     0.324 |        0.704 |
| wavlm_full   | all 13 layers, 20 ms           | si         |      6 |     55.024 |    10.253 |              0.086 |            0.077 |           0 |             6 |     -3.452 |     9.585 |        0.397 |
| wavlm_full   | all 13 layers, 20 ms           | rand       |      5 |     99.857 |     0.213 |              0.088 |            0.077 |           2 |             5 |      0     |     0     |      nan     |
| wavlm_lw0    | layer logits: no weight decay  | sd         |      5 |     99.905 |     0.13  |              0.158 |            0.143 |           0 |             5 |      0     |     0     |      nan     |
| wavlm_lw0    | layer logits: no weight decay  | si         |      6 |     58.417 |     9.971 |              0.156 |            0.143 |           0 |             6 |     -0.06  |     5.331 |        0.978 |
| wavlm_lw0    | layer logits: no weight decay  | rand       |      5 |     99.857 |     0.13  |              0.159 |            0.143 |           2 |             5 |      0     |     0.209 |        1     |
| wavlm_lw0x10 | layer logits: no decay, 10x lr | sd         |      5 |     99.857 |     0.13  |              0.238 |            0.143 |           0 |             5 |     -0.048 |     0.132 |        0.374 |
| wavlm_lw0x10 | layer logits: no decay, 10x lr | si         |      6 |     59.452 |     6.508 |              0.257 |            0.143 |           0 |             6 |      0.976 |     7.571 |        0.754 |
| wavlm_lw0x10 | layer logits: no decay, 10x lr | rand       |      5 |     99.905 |     0.13  |              0.256 |            0.143 |           0 |             5 |      0.048 |     0.324 |        0.704 |
