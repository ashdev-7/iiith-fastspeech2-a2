# english_mfa: v3.1.0 (default) vs. v3.0.0 dictionary

* `mfa_ipa`: english_mfa + v3.1.0 dict (MFA default)
* `mfa_ipa_dict_v300`: english_mfa + v3.0.0 dict (all variants)
* `arpa_base`: english_us_arpa (reference)

## Pronunciations chosen on the ISLE prompts

| experiment | word | phones |
|---|---|---|
| mfa_ipa | said | s aj d |
| mfa_ipa | white | ʋ aj ʈ |
| mfa_ipa | not | n a t |
| mfa_ipa | bait | b e t |
| mfa_ipa | new | ɲ uː |
| mfa_ipa | no | n əw |
| mfa_ipa | bad | b a d |
| mfa_ipa | bed | b ɛ d |
| mfa_ipa_dict_v300 | said | s ɛ d |
| mfa_ipa_dict_v300 | white | w aj t |
| mfa_ipa_dict_v300 | not | n ɔ t |
| mfa_ipa_dict_v300 | bait | b ej t |
| mfa_ipa_dict_v300 | new | ɲ ʉː |
| mfa_ipa_dict_v300 | no | n əw |
| mfa_ipa_dict_v300 | bad | b a d |
| mfa_ipa_dict_v300 | bed | b ɛ d |
| arpa_base | said | S EH1 D |
| arpa_base | white | W AY1 T |
| arpa_base | not | N AA1 T |
| arpa_base | bait | B EY1 T |
| arpa_base | new | N Y UW1 |
| arpa_base | no | N OW1 |
| arpa_base | bad | B AE1 D |
| arpa_base | bed | B EH1 D |

## Last word of each ISLE prompt (where mfa_ipa absorbed the trailing silence)

| experiment | utterance | word | start | end | dur_s | phones | file_silence_pct |
|---|---|---|---|---|---|---|---|
| mfa_ipa | 01 | bait | 1.7 | 4.1 | 2.4 | b e t | 10.4 |
| mfa_ipa | 02 | no | 1.8 | 2.2 | 0.5 | n əw | 57.7 |
| mfa_ipa | 03 | bed | 1.5 | 4.5 | 3.0 | b ɛ d | 4.7 |
| mfa_ipa_dict_v300 | 01 | bait | 1.7 | 4.1 | 2.4 | b ej t | 10.4 |
| mfa_ipa_dict_v300 | 02 | no | 1.8 | 2.2 | 0.5 | n əw | 57.7 |
| mfa_ipa_dict_v300 | 03 | bed | 1.5 | 4.5 | 3.0 | b ɛ d | 4.7 |
| arpa_base | 01 | bait | 1.8 | 2.2 | 0.5 | B EY1 T | 59.8 |
| arpa_base | 02 | no | 1.8 | 2.2 | 0.5 | N OW1 | 59.5 |
| arpa_base | 03 | bed | 1.5 | 2.0 | 0.6 | B EH1 D | 60.2 |

## Word-boundary agreement with arpa_base

| comparison | tier | speaker | boundaries | matched_pct | identical_pct | within_10ms_pct | within_20ms_pct | mean_ms | max_ms |
|---|---|---|---|---|---|---|---|---|---|
| mfa_ipa vs arpa_base | words | f2b | 456 | 100.0 | 41.9 | 82.5 | 88.2 | 14.2 | 270.0 |
| mfa_ipa vs arpa_base | words | isle | 30 | 100.0 | 13.3 | 60.0 | 73.3 | 166.8 | 2460.0 |
| mfa_ipa_dict_v300 vs arpa_base | words | f2b | 456 | 100.0 | 49.8 | 88.6 | 95.2 | 8.2 | 250.0 |
| mfa_ipa_dict_v300 vs arpa_base | words | isle | 30 | 100.0 | 20.0 | 56.7 | 66.7 | 174.8 | 2460.0 |
