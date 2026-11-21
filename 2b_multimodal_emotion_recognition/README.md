# Multimodal Emotion Recognition on TESS (speech · text · fusion)

Three emotion-recognition systems built from the functional blocks in the assignment
(preprocessing → feature extraction → temporal/contextual modelling → [fusion] → classifier),
evaluated on the **Toronto Emotional Speech Set** (2 speakers × 7 emotions × 200 words = 2800 clips).

**Report:** [`report/report.pdf`](report/report.pdf) · **Tables:** [`Results/accuracy_tables.md`](Results/accuracy_tables.md) · **Plots:** [`Results/plots/`](Results/plots)

## Results (176 runs, Kaggle T4 GPU; full log in `logs/kaggle_run_v2.log`)

Test accuracy %, mean ± std over runs (RAND/SD 5 runs, SI 6 runs). Classes are balanced, so accuracy = UAR. Chance = 14.29 %.

| Pipeline | Variant | RAND | SD | SI (unseen speaker) |
|---|---|---|---|---|
| Speech | log-Mel + BiGRU (baseline) | 99.67 ± 0.32 | 99.52 ± 0.56 | 41.57 ± 5.99 |
| Speech | **WavLM + BiGRU** | 99.86 ± 0.21 | 99.90 ± 0.13 | 58.48 ± 8.74 |
| Text | DistilRoBERTa + BiGRU, ground-truth transcript | 13.81 ± 0.56 | 14.29 ± 0.00 | 14.29 ± 0.00 |
| Text | DistilRoBERTa + BiGRU, Whisper transcript | 28.52 ± 1.93 | 30.05 ± 2.42 | 23.46 ± 4.72 |
| Fusion | **Gated GMU**, ground truth | 100.00 ± 0.00 | 99.95 ± 0.11 | 56.77 ± 7.57 |
| Fusion | Gated GMU, Whisper | 99.90 ± 0.13 | 100.00 ± 0.00 | 57.14 ± 7.29 |
| Fusion | Concatenation, ground truth | 99.86 ± 0.13 | 100.00 ± 0.00 | 60.12 ± 11.23 |
| Fusion | Cross-attention, ground truth | 100.00 ± 0.00 | 99.90 ± 0.13 | **63.42 ± 8.84** |

All rows (including speaker-normalised variants and zero-shot text), macro-F1 and per-class F1 are in
[`Results/accuracy_tables.md`](Results/accuracy_tables.md). Paired tests and calibration are in [`Results/statistics.md`](Results/statistics.md).

**Key findings**
* **Random and speaker-dependent splits are saturated** (99.5–100 % for every audio model). Under leave-one-speaker-out, accuracy is 42–63 %.
  Of all the architectural choices, only **WavLM vs. log-Mel** gives a consistent gain (+16.9 pp SI, positive in all 6 runs and both directions;
  paired *t* p = 0.022, but not significant after Holm correction for 24 comparisons, see [`robustness_checks.md`](Results/robustness_checks.md)).
* **Ground-truth text is at chance by construction**: every word occurs once per emotion and speaker, so I(text; emotion) = 0.
  The trained model, a zero-shot emotion classifier and linear probes all confirm it. **Whisper transcripts reach 2.1× chance** because ASR
  punctuation encodes prosody: 72 % of pleasant-surprise transcripts end in "!", and 0 % of sad or neutral ones do.
* **Fusion helps only a little, and only where speech is unsure.** Binned by speech-only confidence, Whisper-text fusion gains +11.8 pp below 0.5,
  but a second speech-only model "gains" +6.1 pp there too (regression to the mean). Net effect: +5.7 pp; none for ground-truth text.
  No fusion variant beats speech-only significantly (cross-attention +4.9 pp, p = 0.39, and it reverses sign between the two transfer directions).
* **The bottleneck is speaker shift: the BiGRU overfits the training speaker** (100 % on its held-out words, 58.5 % on the other speaker).
  A **pre-registered** low-capacity model ([plan](analysis/PREREGISTRATION_si_linear.md), committed before the run) reaches **66.0 % SI**
  (70.6 % with label-free speaker normalisation †): [`si_linear.md`](Results/si_linear.md).
  A post-hoc check over C = 1e-5 … 10 ([`c_sensitivity.md`](Results/c_sensitivity.md)) keeps it at 63.5–68.1 % (66.7–72.7 % †), so it is not an artefact of the tie-broken C.
  A second pre-registered study (80 runs, [`capacity_study.md`](Results/capacity_study.md)): on identical normalised inputs the
  linear head beats the BiGRU by 10–18 pp in both directions (robust to seed noise). Normalisation helps the linear head but not
  the BiGRU, and it works from a 40-word unlabelled enrolment sample (within 0.6 pp of using the whole test set).
  A third pre-registered study ([plan](analysis/PREREGISTRATION_si_augmentation.md), [`augmentation_study.md`](Results/augmentation_study.md)):
  0.9×/1.1× speed augmentation halves the BiGRU's seed spread but does not improve it reliably; the linear head gains
  +1.1/+1.7 pp and reaches **72.0 % SI** †, the best result.
  Label-free per-speaker normalisation also makes SI accuracy **flat under white noise** (56.5 → 54.8 % at 0 dB, vs. 58.5 → 28.0 % without it).
* Easiest emotions: neutral, sad, fear. Hardest: happy and angry, which are confused with each other, with pleasant surprise and with fear (high arousal, different valence).

Additional studies: speech front-end ablation ([`ablation_speech.md`](Results/ablation_speech.md): all 13 WavLM layers at 20 ms gives no gain, -3.5 pp SI, n.s.; freeing the layer weights moves them to the *lowest* layers and does not help SI), hand-crafted 44-feature prosody baseline ([`prosody_baseline.md`](Results/prosody_baseline.md)),
layer-wise probes ([`layer_probe.md`](Results/layer_probe.md)), noise robustness ([`noise_robustness.md`](Results/noise_robustness.md)),
fusion analysis ([`fusion_analysis.md`](Results/fusion_analysis.md)), failure cases ([`error_analysis.md`](Results/error_analysis.md)),
ASR transcript statistics ([`asr_transcript_stats.md`](Results/asr_transcript_stats.md)).


## Architecture

| Block | Speech pipeline | Text pipeline |
|---|---|---|
| Preprocessing | resample 24.4→16 kHz, mono, DC removal, trim silence (30 dB, 50 ms margin), peak-normalise, variable length + masking | lower-case, remove symbols, keep `. , ? !` as tokens, RoBERTa BPE tokenisation |
| Feature extraction | frozen **WavLM-Base+** hidden states (7 layers: 0, 2, …, 12; 40 ms frames) → *time_steps × 7 × 768*. Baseline: 80-d log-Mel | frozen **emotion-tuned DistilRoBERTa** (`j-hartmann/emotion-english-distilroberta-base`) hidden states → *tokens × 7 × 768* |
| Temporal / contextual modelling | learnable softmax layer weighting → LayerNorm → Linear(768→256) → **2-layer BiGRU** → **additive attention pooling** (+ time masking) | layer weighting → Linear(768→256) → **BiGRU** → attention pooling |
| Fusion | **Gated Multimodal Unit** (default): `h = z⊙tanh(W_s s) + (1−z)⊙tanh(W_t t)`, `z = σ(W[s;t])`, trained with modality dropout. Ablations: concatenation, bidirectional cross-modal attention | |
| Classifier | MLP 256 → 128 → 7, label smoothing 0.05 | same |

## Evaluation protocols (no leakage)

* **RAND (random clip-level split, 70/15/15, stratified by speaker × emotion):** the protocol used in most TESS papers. It is optimistic, because the same speakers and words appear in train and test. 5 seeds.
* **SD (speaker-dependent, word-disjoint):** both speakers in every split, but each of the 200 target words
  (all 14 of its recordings) belongs to exactly one split: 140/30/30 words = 1960/420/420 clips. 5 seeds.
* **SI (speaker-independent, leave-one-speaker-out):** train/val on one speaker (170/30 words), test on all
  1400 clips of the **unseen** speaker. 2 directions × 3 seeds.
* The model is selected on validation accuracy (early stopping), and the test set is evaluated once by `test.py`.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate          # (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt                          # GPU: install torch from pytorch.org first
# Kaggle API token in ~/.kaggle (kaggle.json or access_token)
python prepare_data.py --all     # download TESS, index, preprocess, cache WavLM / log-Mel / text features, Whisper ASR
```

Paths can be changed with `MER_DATA_DIR` (raw data, default `~/data/tess`), `MER_CACHE_DIR` (features, ~2 GB,
default `~/data/tess_cache`) and `MER_RUNS_DIR` (checkpoints, default `./runs`).

## Train / test

```bash
python models/speech_pipeline/train.py                 # all SD + SI runs; --features logmel for the baseline
python models/speech_pipeline/test.py
python models/text_pipeline/train.py                   # --transcripts gt | asr
python models/text_pipeline/test.py
python models/fusion_pipeline/train.py                 # --fusion gated | concat | xattn   --transcripts gt | asr
python models/fusion_pipeline/test.py
# single run: add  --protocol si --fold 0 --seed 0
python analysis/make_results.py                        # tables + plots -> Results/
python analysis/error_analysis.py                      # failure cases -> Results/error_analysis.md
python analysis/layer_probe.py                         # linear probes per frozen layer
python analysis/prosody_baseline.py                    # hand-crafted prosody baseline
python analysis/noise_robustness.py                    # test-time noise (needs checkpoints + GPU)
python run_all.py --skip-prepare --skip-train --skip-analyses --ablations   # speech front-end ablation (48 runs)
python analysis/si_linear.py                           # pre-registered SI experiment (CPU, cached features)
python analysis/c_sensitivity.py                       # post-hoc: how much the linear model depends on C
python analysis/robustness_checks.py                   # Holm correction, per-direction tests, regression-to-mean control
python analysis/capacity_study.py                      # pre-registered head-capacity study (runs_capacity/, Kaggle job 'capacity')
python tests/test_invariants.py && python tests/report_tables_check.py   # leakage/alignment tests; table cells vs data
python run_all.py                                      # everything above in one go
```

## Layout

```
common/           config, data/splits, audio preprocessing, feature extraction + store, model blocks, train/eval engine
models/{speech,text,fusion}_pipeline/{train,test}.py
analysis/         make_results.py, error_analysis.py, layer_probe.py, prosody_baseline.py, noise_robustness.py, recover_from_log.py,
                  si_linear.py (+ PREREGISTRATION_si_linear.md), robustness_checks.py
Results/          accuracy tables (md/csv), significance + calibration, per-class F1, analysis tables, plots/
runs/             all 176 runs: train log, test metrics, predictions, embeddings (checkpoints not included)
artifacts/        Whisper transcripts (clean + noisy), zero-shot text predictions, clip index, exported from the feature cache
logs/             full logs of the Kaggle runs that produced every number (main run, ablation, analyses re-run)
kaggle/           make_kernel.py: builds the Kaggle GPU jobs (full / ablation / analyses) that produced them
report/           report.tex / report.pdf
prepare_data.py   run_all.py   requirements.txt
```

## Provenance note

Every result was produced by one Kaggle GPU job running `prepare_data.py --speech --text --asr` and then `run_all.py`.
In that job, `layer_probe.py`, `prosody_baseline.py` and `noise_robustness.py` computed and printed all their per-run values but crashed
when saving, because `Results/` did not yet exist (now fixed in `common/config.py`).

* `layer_probe.py` and `prosody_baseline.py` were then **re-run** in a separate Kaggle job (`logs/kaggle_run_analyses.log`,
  `python kaggle/make_kernel.py analyses`); their files in `Results/` are those genuine outputs. They agree with the values
  printed in the first job (prosody: identical; probes: within 0.05 pp, i.e. the 3-decimal print rounding).
* `noise_robustness.py` needs the 66 trained checkpoints, which that job did not keep. Its results were first rebuilt from the
  printed per-run values (`analysis/recover_from_log.py`), then **regenerated** in their own job (`python kaggle/make_kernel.py noise`,
  `logs/kaggle_run_noise.log`), with checks registered in advance (`analysis/PREREGISTRATION_si_augmentation.md`, part N):
  all 66 retrained runs give 100 % identical predictions to `runs/`, every clean and noisy Whisper transcript is identical,
  and all 330 per-run noise accuracies match the log values to within 0.05 pp (`Results/noise_regeneration_check.md`).
  The files in `Results/` are now the regenerated outputs.

Kaggle jobs are built with `kaggle/make_kernel.py` (`full`, `ablation`, `analyses`, `capacity`, `augment`, `noise`).
