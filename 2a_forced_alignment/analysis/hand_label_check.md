# Hand-labelled reference (two blind passes)

60 words, 120 boundaries (`hand_labels/targets.csv`).

## Annotator consistency (precision of the reference)

|pass1 - pass2|: median 11.3 ms, mean 35.2 ms, 63 % within 20 ms, max 195 ms.

## Aligner error against the reference (reference = mean of the two passes)

| experiment        |   boundaries |   median_abs_ms |   mean_abs_ms |   within_20ms |   within_50ms |   max_abs_ms |
|:------------------|-------------:|----------------:|--------------:|--------------:|--------------:|-------------:|
| arpa_base         |          120 |            24.5 |          33.1 |          42.5 |          75.8 |        166.4 |
| arpa_adapted      |          120 |            26.3 |          33.1 |          42.5 |          76.7 |        166.4 |
| arpa_g2p_custom   |          120 |            22.5 |          34.1 |          43.3 |          75.8 |        177.4 |
| mfa_ipa           |          120 |            25.7 |          75.7 |          37.5 |          69.2 |       2435.6 |
| mfa_ipa_dict_v300 |          120 |            26   |          71.1 |          40.8 |          71.7 |       2435.6 |

|                               |   boundaries |   median_abs_ms |   mean_abs_ms |   within_20ms |   within_50ms |   max_abs_ms |
|:------------------------------|-------------:|----------------:|--------------:|--------------:|--------------:|-------------:|
| ('arpa_base', 'f2b')          |           90 |            22.9 |          31.7 |          44.4 |          74.4 |        115.8 |
| ('arpa_base', 'isle')         |           30 |            28.9 |          37.6 |          36.7 |          80   |        166.4 |
| ('arpa_adapted', 'f2b')       |           90 |            23.8 |          31.7 |          44.4 |          75.6 |        115.8 |
| ('arpa_adapted', 'isle')      |           30 |            28.9 |          37.6 |          36.7 |          80   |        166.4 |
| ('arpa_g2p_custom', 'f2b')    |           90 |            20.8 |          30.8 |          46.7 |          75.6 |        115.8 |
| ('arpa_g2p_custom', 'isle')   |           30 |            29.3 |          43.9 |          33.3 |          76.7 |        177.4 |
| ('mfa_ipa', 'f2b')            |           90 |            27.9 |          39.9 |          38.9 |          70   |        237.8 |
| ('mfa_ipa', 'isle')           |           30 |            21.3 |         183   |          33.3 |          66.7 |       2435.6 |
| ('mfa_ipa_dict_v300', 'f2b')  |           90 |            23.3 |          31.4 |          44.4 |          74.4 |        105.8 |
| ('mfa_ipa_dict_v300', 'isle') |           30 |            29.7 |         190.2 |          30   |          63.3 |       2435.6 |

## Paired against arpa_base (same boundaries)

| experiment        |   mean_extra_error_ms |   boundaries_worse |   boundaries_better |   p_wilcoxon |
|:------------------|----------------------:|-------------------:|--------------------:|-------------:|
| arpa_adapted      |                  -0   |                  1 |                   1 |   nan        |
| arpa_g2p_custom   |                   0.9 |                 11 |                   7 |     0.861496 |
| mfa_ipa           |                  42.5 |                 42 |                  38 |     0.261159 |
| mfa_ipa_dict_v300 |                  37.9 |                 35 |                  36 |     0.763231 |
