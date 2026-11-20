# Aligner-independent reference: pauses found from the Praat intensity contour

Pauses are found from Praat's intensity contour (`Sound: To Intensity`, 100 Hz, 5 ms step): stretches of >= 150 ms whose intensity stays within 6, 9, 12 dB of the file's noise floor (10th percentile), bridging sounding blips < 50 ms. No aligner or human is involved.

* `edges`: speech offsets/onsets at the detected pauses; error = aligner's nearest word end/start minus the Praat landmark.
* `pauses_kept_pct`: share of detected pauses that the aligner leaves (mostly) outside words; `pause_time_in_words_pct`: mean share of pause time the aligner assigns to words.

**Precision of the reference:** a landmark found at several thresholds moves by a median of 5 ms (90th percentile 69 ms, n = 50) between them, because the energy threshold decides where a fading word ends. The edge errors also change a lot with the threshold, since an energy detector calls breaths, clicks and noise bursts "sound". **Use this reference for pause absorption only** (`pause_time_in_words_pct`, stable across thresholds), not to rank aligners at the 10-20 ms level.

|                                     |   edges |   median_abs_ms |   mean_abs_ms |   within_20ms |   within_50ms |   max_abs_ms |   pauses |   pauses_kept_pct |   pause_time_in_words_pct |
|:------------------------------------|--------:|----------------:|--------------:|--------------:|--------------:|-------------:|---------:|------------------:|--------------------------:|
| (6.0, 'f2b', 'arpa_base')           |      48 |            37.5 |         106.7 |          37.5 |          54.2 |        955.9 |       25 |            84     |                      24.9 |
| (6.0, 'f2b', 'arpa_adapted')        |      48 |            37.5 |         106.9 |          35.4 |          54.2 |        955.9 |       25 |            84     |                      25.1 |
| (6.0, 'f2b', 'arpa_g2p_custom')     |      48 |            37.5 |         106.7 |          37.5 |          54.2 |        955.9 |       25 |            84     |                      24.7 |
| (6.0, 'f2b', 'mfa_ipa')             |      48 |            29.8 |         103.4 |          35.4 |          60.4 |        935.9 |       25 |            92     |                      21.9 |
| (6.0, 'f2b', 'mfa_ipa_dict_v300')   |      48 |            35.7 |         104.8 |          29.2 |          56.2 |        935.9 |       25 |            92     |                      24.4 |
| (6.0, 'isle', 'arpa_base')          |      14 |           525   |         809.3 |          28.6 |          28.6 |       2370   |        9 |           100     |                       0.1 |
| (6.0, 'isle', 'arpa_adapted')       |      14 |           525   |         809.3 |          28.6 |          28.6 |       2370   |        9 |           100     |                       0.1 |
| (6.0, 'isle', 'arpa_g2p_custom')    |      14 |           525   |         809.3 |          28.6 |          28.6 |       2370   |        9 |           100     |                       0.1 |
| (6.0, 'isle', 'mfa_ipa')            |      14 |           582.5 |         721.8 |          21.4 |          28.6 |       2390   |        9 |            44.444 |                      56.2 |
| (6.0, 'isle', 'mfa_ipa_dict_v300')  |      14 |           582.5 |         721.8 |          21.4 |          28.6 |       2390   |        9 |            44.444 |                      56.2 |
| (9.0, 'f2b', 'arpa_base')           |      70 |            39.3 |          89   |          34.3 |          51.4 |        955.9 |       36 |            75     |                      37.6 |
| (9.0, 'f2b', 'arpa_adapted')        |      70 |            39.3 |          89.2 |          32.9 |          51.4 |        955.9 |       36 |            75     |                      37.7 |
| (9.0, 'f2b', 'arpa_g2p_custom')     |      70 |            39.3 |          89.6 |          34.3 |          51.4 |        955.9 |       36 |            75     |                      38   |
| (9.0, 'f2b', 'mfa_ipa')             |      70 |            37.5 |          89.2 |          31.4 |          55.7 |        935.9 |       36 |            75     |                      36.4 |
| (9.0, 'f2b', 'mfa_ipa_dict_v300')   |      70 |            42.7 |          89.2 |          25.7 |          52.9 |        935.9 |       36 |            75     |                      38.3 |
| (9.0, 'isle', 'arpa_base')          |       8 |            50   |         181.9 |          50   |          50   |        660   |        6 |           100     |                       0.2 |
| (9.0, 'isle', 'arpa_adapted')       |       8 |            50   |         181.9 |          50   |          50   |        660   |        6 |           100     |                       0.2 |
| (9.0, 'isle', 'arpa_g2p_custom')    |       8 |            50   |         181.9 |          50   |          50   |        660   |        6 |           100     |                       0.2 |
| (9.0, 'isle', 'mfa_ipa')            |       8 |           277.5 |         329.4 |          37.5 |          50   |        745   |        6 |            66.667 |                      34.6 |
| (9.0, 'isle', 'mfa_ipa_dict_v300')  |       8 |           277.5 |         329.4 |          37.5 |          50   |        745   |        6 |            66.667 |                      34.6 |
| (12.0, 'f2b', 'arpa_base')          |      80 |            44.3 |          78.2 |          28.7 |          52.5 |        955.9 |       41 |            65.854 |                      45.2 |
| (12.0, 'f2b', 'arpa_adapted')       |      80 |            44.3 |          78.4 |          27.5 |          52.5 |        955.9 |       41 |            65.854 |                      45.3 |
| (12.0, 'f2b', 'arpa_g2p_custom')    |      80 |            44.3 |          78.7 |          28.7 |          52.5 |        955.9 |       41 |            65.854 |                      45.5 |
| (12.0, 'f2b', 'mfa_ipa')            |      80 |            38.9 |          78.7 |          26.2 |          56.2 |        935.9 |       41 |            63.415 |                      45.8 |
| (12.0, 'f2b', 'mfa_ipa_dict_v300')  |      80 |            42.5 |          79.6 |          22.5 |          52.5 |        935.9 |       41 |            60.976 |                      47.3 |
| (12.0, 'isle', 'arpa_base')         |       7 |             5   |         115.7 |          57.1 |          57.1 |        660   |        6 |           100     |                       0.9 |
| (12.0, 'isle', 'arpa_adapted')      |       7 |             5   |         115.7 |          57.1 |          57.1 |        660   |        6 |           100     |                       0.9 |
| (12.0, 'isle', 'arpa_g2p_custom')   |       7 |             5   |         115.7 |          57.1 |          57.1 |        660   |        6 |           100     |                       0.9 |
| (12.0, 'isle', 'mfa_ipa')           |       7 |           110   |         284.3 |          42.9 |          42.9 |        670   |        6 |            66.667 |                      35.8 |
| (12.0, 'isle', 'mfa_ipa_dict_v300') |       7 |           110   |         284.3 |          42.9 |          42.9 |        670   |        6 |            66.667 |                      35.8 |
