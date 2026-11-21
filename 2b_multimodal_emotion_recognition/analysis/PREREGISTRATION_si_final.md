# Pre-registration 5: the final speech model for the unseen speaker (clean levers only)

Written and pushed to GitHub **before** the Kaggle job was started; the git server timestamp of this file
is the proof of order. The analysis code (`analysis/final_features.py`, `analysis/final_study.py`) is
committed together with this plan.

**What has been run before this plan (full disclosure).**
* `final_study.py` on random (null) features: every accuracy is at chance (13–16 %, chance = 14.3 %), and no test is significant.
* `final_study.py` on features with a planted label signal: the planted gain is detected (H1 replicated).
* `final_features.py` on 2–3 clips with `openai/whisper-tiny`: shapes, finiteness, and the pitch-shifted copies
  differ from the originals. No labels are involved.

No TESS label has been used to choose anything in this plan. The only test-speaker results seen so far are
those of studies 1–4, which are already in the report.

## Rules (from the assignment and from studies 1–4; none is relaxed here)

The assignment names TESS as the dataset, so the only labelled data is TESS. Pretrained encoders are used
frozen, and only if they were trained **without emotion labels**. Excluded, explicitly:
* any model trained with emotion labels whose data may include TESS (all emotion2vec+ models, hub
  "speech emotion recognition" checkpoints);
* any other labelled emotion corpus;
* any use of TESS's recording design at test time:
  * no per-word assignment (each word appears in all 7 emotions);
  * no class-balance prior (each emotion is 1/7 of the test set);
* any choice made on the test speaker's labels.

## Question

Can three clean levers, each motivated before any test, improve the best clean model of study 4 (A-L1n with
frozen WavLM-Large, 75.7 % †)?
* a stronger frozen encoder;
* augmentation aimed at the speaker shift;
* label-free adaptation to the new speaker.

## The three levers (fixed)

1. **W3, Whisper-large-v3 encoder** (`openai/whisper-large-v3`, Radford et al., ICML 2023), frozen and used
   encoder-only.
   * **Features:** hidden states 0, 2, …, 32 (17 layers × 1280). Whisper pads every clip to 30 s; only the
     encoder frames that cover the clip itself are kept, ceil(samples / 320) at 20 ms each. They are pooled
     exactly like WavLM: frame pairs averaged (40 ms), then mean + std per layer. It runs in float16 on the GPU,
     and every feature is checked to be finite.
   * **Why:** on the EmoBox cross-corpus benchmark (Ma et al., Interspeech 2024), Whisper-large-v3 beats WavLM-Large on most
     train→test corpus pairs.
   * **Leakage:** Whisper is trained to output transcripts only, on about 5 M hours of web audio. Even if TESS
     audio were among it, the target for every TESS clip would be "Say the word …", which is identical across
     emotions. No emotion label can have been learned. This is the same exposure class as WavLM.
2. **P, pitch-shift augmentation.** Four extra copies of every training clip are added, shifted by −4, −2, +2
   and +4 semitones with tempo kept (`librosa.effects.pitch_shift`, then peak-normalised to 0.95).
   * **Why:** the two speakers are 64 and 26 years old (TESS description), and voice pitch is the most visible
     difference between them. The speed copies of study 3 change pitch by at most ±1.7 semitones and also change
     tempo.
   * As before, each copy is z-scored as its own pseudo-speaker, and the test speaker's copies are never read.
3. **S, self-training on the unseen speaker's unlabelled clips** (pseudo-labelling; Lee, ICML-W 2013).
   * Starting from the trained model, three rounds each take the 25 %, 50 % and then 75 % of the pool clips the
     model is most confident about (maximum probability).
   * Those clips are labelled with the model's own prediction, and the model is refit on the training rows plus
     those clips (original audio only).
   * There is no class quota and no use of any class prior. The test labels are never read.
   * In the transductive variant (†) the pool is all of the test speaker's clips, in the same class as the
     per-speaker normalisation used since study 2.
   * In the inductive variant (I) the pool is only the 40 calibration words (numpy seed 7, as in studies 2–4),
     and accuracy is scored on the other 160 words. Those 160 words are never in the pool (audit D5).

Every model uses the unchanged A-L1n recipe of studies 3–4: pooled mean + std of every kept layer, z-scoring per
(pseudo-)speaker, StandardScaler, then L2 logistic regression with **C = 0.001**, with no re-tuning.

## Models (all fixed)

| name | definition |
|---|---|
| R = A-L1n(WL) | reference: study 4's best clean model (WavLM-Large, original + 0.9×/1.1× speed copies) |
| A-L1n(W3) | the same recipe on Whisper-large-v3 |
| AP-L1n(WL), AP-L1n(W3) | + the four pitch copies |
| ENS2 | equal-weight mean of the probabilities of AP-L1n(WL) and AP-L1n(W3) |
| ST(R) | self-training applied to R |
| **FINAL (primary)** | self-training applied to ENS2: both members are refit on the same ensemble pseudo-labels each round |

## Hypotheses

| id | difference (per direction) | prediction |
|---|---|---|
| H1 (primary) | FINAL − R | > 0 in both directions |
| H2-W | A-L1n(W3) − R | none (two-sided) |
| H2-P | AP-L1n(WL) − R | none (two-sided) |
| H2-S | ST(R) − R | none (two-sided) |
| H3 | inductive FINAL − transductive FINAL, on the same 160 words | ≥ −2 pp |

**Reporting rule (strict).** If H1 is replicated (+), FINAL becomes the report's best SI system: † is the
headline and the inductive result is reported next to it. Otherwise R stays the best system. **No other model is
promoted, whatever it scores**, and every row is reported.

## Audits and checks (all must pass)

* **D1–D4** as in study 4 (word identity carries no label information; speakers and clips are disjoint;
  normalisation sees features only; the test speaker's perturbed copies are never read).
* **D5.** In the inductive variant, no scored clip is ever pseudo-labelled.
* **D6.** The fitting code receives only the training speaker's labels. Test labels are read once, after
  every prediction is fixed.
* **C1.** R recomputed in this job gives ≥ 99 % identical predictions to study 4
  (`Results/backbone_predictions.csv`). The tolerance allows for GPU numerics, because the WavLM-Large features are
  re-extracted. If C1 fails, the job stops before any test is computed.

## Statistics

* Word-cluster bootstrap over the 200 test words (10 000 resamples, numpy seed 0). All models are deterministic.
* Verdicts as in studies 2–4: replicated / direction-specific / equivalent within ±2 pp / inconclusive.
* Holm correction over H1 + H2: 4 comparisons × 2 directions = 8 tests.
* Post hoc, labelled as such: the accuracy of the pseudo-labels in each self-training round. It is computed
  after the predictions are fixed and is never used to choose.
* With two speakers, agreement between the two directions is descriptive, not a claim about speakers in general.

## Amendment 1 (after the first run; nothing in the plan above is changed)

Run 1 (`logs/kaggle_run_final_v1_failed.log`) extracted every feature and was then killed by the operating
system during `final_study.py`: it ran out of memory (exit −9) while fitting FINAL in the first direction.
The step prints only "fitted" / "done" lines, so **no accuracy, prediction or test was computed or seen**.

The only changes are about memory:
* The analysis now holds one encoder/recipe pair in memory at a time (the two FINAL members together).
* The classifier runs step by step: the scaler works in place, followed by the float64 conversion that
  LogisticRegression's lbfgs input check performs anyway.
* The arithmetic is unchanged. On the null and planted-signal validation features, the new code gives 100 %
  identical predictions (transductive and inductive, all 19 600 rows) and identical test statistics to the
  committed version.
* The job now keeps the features in its output, so that a further failure of the analysis would not require
  re-extraction, and it prints the machine's RAM.

The job is re-run unchanged otherwise.

## Amendment 2 (after runs 2 and 3; nothing in the plan above is changed)

* **Run 2** (Kaggle version 2) stayed queued and never started, so it produced nothing. It was resubmitted
  unchanged as version 3.
* **Run 3** (`logs/kaggle_run_final_v3_failed.log`) failed the same way as run 1. All features were extracted
  and saved, and `final_study.py` was then killed for running out of memory (exit −9) at the same point, on a
  33.7 GB machine. Again it printed only "fitted" / "done" lines, so **no accuracy, prediction or test was
  computed or seen**.
* **Diagnosis.** Both runs died after about 14 model fits, although they held different amounts of data
  in memory. This points to memory accumulating across fits within one process, not to a single large peak.
* **Changes** (packaging and memory only; the arithmetic is unchanged):
  1. Each analysis stage of each direction runs in a fresh process, which returns all of its memory to the
     operating system when it ends. The stages are R + ST(R), A-L1n(W3), FINAL †, and FINAL inductive.
  2. Each base model is fitted once and predicts both test variants. Before, it was fitted twice on
     identical data, which gives an identical deterministic model.
  3. Memory use is printed after every fit.
* **Verification.** On the null and planted-signal validation features, the predictions (19 600 rows,
  transductive and inductive), the test table and the self-training diagnostic are all 100 % identical to
  the committed version.
* **Run 4** (job `final_analysis`, CPU only) runs only the analysis, on the features saved by run 3, which are
  attached as a kernel source and not re-extracted.
