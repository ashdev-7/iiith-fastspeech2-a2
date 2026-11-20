# Pre-registration: a low-capacity classifier for the speaker-independent (SI) protocol

Written and committed **before** the experiment was run on the test speaker (see the git history:
this file is committed in a separate, earlier commit than `si_linear.py`'s results).

## Motivation (from results already in the report)
* WavLM + BiGRU: 99.95 % SD but 58.5 % SI (std 5–11 pp across seeds). The model fits the single
  training speaker almost perfectly and transfers poorly: overfitting to the *speaker*, not to the clips.
* Per-layer linear probes reach 69.5–71.7 % SI at L8–L10. **That number is optimistic**: the layer was
  read off the test-speaker results. It motivates the hypothesis but cannot be reported as a result.

## Hypothesis (H1)
A low-capacity model whose every choice is fixed in advance or selected on the training speaker only
generalises to the unseen speaker better than WavLM + BiGRU (58.5 % SI).

## Primary model (fixed now, not changed after seeing test results)
* Features: for each clip, mean and std over time of **all 7 cached WavLM-Base+ layers** (L0, L2, …, L12),
  concatenated (7 × 1536 = 10 752 dims). No layer selection.
* Classifier: StandardScaler (fitted on training-speaker clips) + L2 multinomial logistic regression.
* Hyper-parameter: C ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1}, chosen by 5-fold cross-validation
  **grouped by word, on the training speaker only**. The test speaker is never seen during selection.
* Protocol: the existing SI splits (train OAF → test YAF, train YAF → test OAF). Deterministic: one
  run per direction.

## Pre-declared secondary model
* Same, plus label-free per-speaker standardisation of the features (each speaker z-scored with its own
  statistics; the test speaker's statistics come from its unlabelled audio). Transductive, marked †,
  exactly as the existing speaker-norm rows in the report.

## Analysis plan
* Report SI accuracy and macro-F1 per direction and the mean over the two directions.
* Compare with WavLM + BiGRU on the same 2800 test clips: McNemar test against each of its 6 SI runs, and
  the mean difference. With two directions only, no t-test across directions is meaningful.
* Sanity check: the per-layer probes must reproduce `Results/layer_probe.csv` on this feature store.
* Whatever the outcome, both models are reported. No other variant is evaluated on the test speaker in
  this experiment; anything else would be labelled exploratory.

## Known limitation, stated in advance
Selection on the training speaker cannot measure transfer to a new speaker. With two speakers in TESS,
any SI result is two data points; conclusions are about this corpus.
