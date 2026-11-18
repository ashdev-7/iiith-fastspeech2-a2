# Sanity checks

## 1. Audio identity (prepared corpus vs. provided files)

All files sample-identical: **True**

| file | provided_format | converted_rate | converted_channels | converted_bytes_per_sample | converted_samples | header_rate | header_samples | header_max | header_min | converted_max | converted_min | samples_identical |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F2BJRLP1 | NIST SPHERE (pcm, 16-bit LE) | 16000 | 1 | 2 | 404946 | 16000.0 | 404946.0 | 8362.0 | -8740.0 | 8362.0 | -8740.0 | True |
| F2BJRLP2 | NIST SPHERE (pcm, 16-bit LE) | 16000 | 1 | 2 | 458359 | 16000.0 | 458359.0 | 8530.0 | -8427.0 | 8530.0 | -8427.0 | True |
| F2BJRLP3 | NIST SPHERE (pcm, 16-bit LE) | 16000 | 1 | 2 | 491309 | 16000.0 | 491309.0 | 9284.0 | -10295.0 | 9284.0 | -10295.0 | True |
| ISLE_SESS0131_BLOCKD02_01_sprt1 | RIFF/WAVE | 16000 | 1 | 2 | 66000 | nan | nan | nan | nan | nan | nan | True |
| ISLE_SESS0131_BLOCKD02_02_sprt1 | RIFF/WAVE | 16000 | 1 | 2 | 62000 | nan | nan | nan | nan | nan | nan | True |
| ISLE_SESS0131_BLOCKD02_03_sprt1 | RIFF/WAVE | 16000 | 1 | 2 | 72000 | nan | nan | nan | nan | nan | nan | True |

## 2. Words that differ when the provided files are aligned untouched

`spn` = MFA's spoken-noise phone, used for words not in the dictionary (no phone-level alignment).

| file | start | end | prepared_words | untouched_words | prepared_phones | untouched_phones |
|---|---|---|---|---|---|---|
| F2BJRLP1 | 20.6 | 21.4 | w b u r's | wbur's | D AH1 B AH0 L Y UW0 B IY1 Y UW1 AA1 R Z | spn AH0 Z |
| F2BJRLP2 | 5.4 | 6.5 | depoliticize | de politicize | D EH2 P AH0 L IH1 T IH0 S AY2 Z | D IY1 spn |

## 3. Boundary agreement, with MFA's run-to-run noise floor

Boundaries of label-matched intervals (silences excluded; for the untouched input, the spans of the words listed in section 2 are excluded, so only words with identical text are compared). `matched_pct` = share of reference intervals whose label sequence is matched. 10 ms = one MFA frame.

| comparison | tier | speaker | boundaries | matched_pct | identical_pct | within_10ms_pct | within_20ms_pct | mean_ms | max_ms |
|---|---|---|---|---|---|---|---|---|---|
| repeat: arpa_base vs run1 | words | f2b | 456 | 100.0 | 99.3 | 100.0 | 100.0 | 0.1 | 10.0 |
| repeat: arpa_base vs run1 | words | isle | 30 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| repeat: arpa_base vs run1 | phones | f2b | 2050 | 100.0 | 99.2 | 99.9 | 100.0 | 0.1 | 20.0 |
| repeat: arpa_base vs run1 | phones | isle | 76 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| repeat: arpa_base vs run2 | words | f2b | 456 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| repeat: arpa_base vs run2 | words | isle | 30 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| repeat: arpa_base vs run2 | phones | f2b | 2050 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| repeat: arpa_base vs run2 | phones | isle | 76 | 100.0 | 94.7 | 100.0 | 100.0 | 0.5 | 10.0 |
| repeat: run1 vs run2 | words | f2b | 456 | 100.0 | 99.3 | 100.0 | 100.0 | 0.1 | 10.0 |
| repeat: run1 vs run2 | words | isle | 30 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| repeat: run1 vs run2 | phones | f2b | 2050 | 100.0 | 99.2 | 99.9 | 100.0 | 0.1 | 20.0 |
| repeat: run1 vs run2 | phones | isle | 76 | 100.0 | 94.7 | 100.0 | 100.0 | 0.5 | 10.0 |
| untouched input vs arpa_base | words | f2b | 444 | 100.0 | 98.4 | 100.0 | 100.0 | 0.2 | 10.0 |
| untouched input vs arpa_base | words | isle | 30 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| untouched input vs arpa_base | phones | f2b | 1998 | 100.0 | 99.1 | 99.8 | 99.9 | 0.1 | 30.0 |
| untouched input vs arpa_base | phones | isle | 76 | 100.0 | 97.4 | 100.0 | 100.0 | 0.3 | 10.0 |
| arpa_adapted vs arpa_base | words | f2b | 456 | 100.0 | 99.3 | 100.0 | 100.0 | 0.1 | 10.0 |
| arpa_adapted vs arpa_base | words | isle | 30 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| arpa_adapted vs arpa_base | phones | f2b | 2050 | 100.0 | 99.3 | 99.9 | 100.0 | 0.1 | 20.0 |
| arpa_adapted vs arpa_base | phones | isle | 76 | 100.0 | 97.4 | 100.0 | 100.0 | 0.3 | 10.0 |
| arpa_g2p_custom vs arpa_base | words | f2b | 456 | 100.0 | 85.1 | 93.9 | 96.5 | 4.7 | 250.0 |
| arpa_g2p_custom vs arpa_base | words | isle | 30 | 100.0 | 86.7 | 90.0 | 96.7 | 7.0 | 160.0 |
| arpa_g2p_custom vs arpa_base | phones | f2b | 1868 | 91.0 | 94.3 | 98.4 | 98.8 | 1.6 | 250.0 |
| arpa_g2p_custom vs arpa_base | phones | isle | 70 | 92.1 | 87.1 | 95.7 | 98.6 | 3.7 | 160.0 |
