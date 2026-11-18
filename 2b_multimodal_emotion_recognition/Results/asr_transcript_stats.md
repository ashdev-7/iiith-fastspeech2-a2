# Whisper-small transcripts of TESS: what survives into text

% of clips per emotion whose transcript ends in ?, !, ., contains the correct target word, or is exactly the carrier sentence.

| emotion   |   ends_q |   ends_excl |   ends_period |   word_correct |   exact_carrier |
|:----------|---------:|------------:|--------------:|---------------:|----------------:|
| angry     |      0   |         9.2 |          90.8 |           90.5 |            85   |
| disgust   |      0   |         4.5 |          95.5 |           91.8 |            84   |
| fear      |      2.8 |        32   |          65   |           83.5 |            59   |
| happy     |      0   |        42.8 |          57.2 |           86.2 |            79.8 |
| neutral   |      0.5 |         0   |          87.2 |           89.8 |            63.5 |
| ps        |      0.2 |        72.2 |          27.5 |           89.5 |            86.2 |
| sad       |      0   |         0   |          99.5 |           84.8 |            70.8 |

Examples:

- **Angry**: `Say the word back.` | `Say the word bar.` | `Say the word base.` | `Say the word bath.`
- **Disgust**: `Say the word back.` | `Say the word bar.` | `Say the word base.` | `Say the word bath.`
- **Fear**: `Say the word back!` | `Say the word, Bar.` | `Say the word base?` | `Say the word bath?`
- **Happy**: `Say the word back!` | `Say the word bar.` | `Say the word base!` | `Say the word bath!`
- **Neutral**: `Say the word back.` | `Say the word, Bar.` | `Say the word, base.` | `Say the word, Bath.`
- **Pleasant surprise**: `Say the word back!` | `Say the word bar!` | `Say the word base!` | `Say the word bath!`
- **Sad**: `Say the word back.` | `Say the word, bar.` | `Say the word base.` | `Say the word bath.`