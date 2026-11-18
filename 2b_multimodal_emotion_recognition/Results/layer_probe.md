# Linear probes on frozen-encoder layers (pooled mean+std, logistic regression)

emotion_sd / emotion_si: 7-class accuracy; emotion_si_spknorm: features z-normalised per speaker (label-free); speaker_id: OAF-vs-YAF accuracy (SD splits).

| store   | layer            |   emotion_sd |   emotion_si |   emotion_si_spknorm |   speaker_id |
|:--------|:-----------------|-------------:|-------------:|---------------------:|-------------:|
| wavlm   | WavLM L0         |         99.9 |         47.2 |                 57.5 |          100 |
| wavlm   | WavLM L2         |        100   |         51.9 |                 63.6 |          100 |
| wavlm   | WavLM L4         |        100   |         62   |                 67   |          100 |
| wavlm   | WavLM L6         |        100   |         63.6 |                 70.5 |          100 |
| wavlm   | WavLM L8         |        100   |         69.5 |                 73.5 |          100 |
| wavlm   | WavLM L10        |         99.7 |         71.7 |                 71.8 |          100 |
| wavlm   | WavLM L12        |        100   |         56.5 |                 62.7 |          100 |
| logmel  | log-Mel          |         99.3 |         48.7 |                 46.6 |          100 |
| text_gt | DistilRoBERTa L0 |         14.3 |         14.3 |                 14.3 |           50 |
| text_gt | DistilRoBERTa L1 |         14.3 |         14.3 |                 14.3 |           50 |
| text_gt | DistilRoBERTa L2 |         14.3 |         14.3 |                 14.3 |           50 |
| text_gt | DistilRoBERTa L3 |         14.3 |         14.3 |                 14.3 |           50 |
| text_gt | DistilRoBERTa L4 |         14.3 |         14.3 |                 14.3 |           50 |
| text_gt | DistilRoBERTa L5 |         14.3 |         14.3 |                 14.3 |           50 |
| text_gt | DistilRoBERTa L6 |         14.3 |         14.3 |                 14.3 |           50 |
