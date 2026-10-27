# UR-FUNNY V1 application experiment

AdaTrustMSA's feature adapter completed training, validation and evaluation on
the official UR-FUNNY V1 folds on an NVIDIA RTX 4090. The context model achieved
**68.84% accuracy and 0.6884 Macro-F1 on all 3,290 test samples**. These are
measurements from this project, not results copied from the official paper.

## Task and data

The task is binary humor classification: 0 is non-humorous and 1 is humorous.
A possible research application is prioritising humorous utterances in a
speech/video review tool. No deployed editing system or sentiment-classification
result is established by this experiment.

The [official V1 feature release](https://github.com/ROC-HCI/UR-FUNNY/blob/master/UR-FUNNY-V1.md)
and its unmodified `data_folds.pkl` were used. Sample IDs are unique and disjoint
across partitions; provided modality IDs and within-sentence word counts agree.

| Partition | Samples | Non-humorous | Humorous |
|---|---:|---:|---:|
| Training | 10,598 | 5,292 | 5,306 |
| Validation | 2,626 | 1,313 | 1,313 |
| Test | 3,290 | 1,652 | 1,638 |

Text uses 300-dimensional GloVe vectors, audio uses 81-dimensional COVAREP
features, and vision uses 75-dimensional OpenFace features. Word features are
averaged within each sentence. The context model retains the last eight context
sentences, in order, followed by the final utterance. BiGRU encoders and new
binary heads feed the existing AdaTrustMSA fusion core. Training starts from
scratch; no MOSEI checkpoint, BERT text encoder or raw-video extractor is used.

## Training and checkpoint selection

| Setting | Value |
|---|---|
| Random seed | 2026; one seed |
| Optimizer | AdamW |
| Learning rate | 0.0007 |
| Weight decay | 0.01 |
| Batch size | 128 |
| Dropout | 0.2 |
| Maximum epochs | 30 |
| Early stopping | Patience 6 |
| Checkpoint selection | Validation Macro-F1 only |
| Context model | 8 actual epochs; selected epoch 2 |
| Final-utterance-only model | 10 actual epochs; selected epoch 4 |

Imputation and standardisation are fitted on training observations only,
excluding padding. Standardised features are clipped to [-5, 5]. Training uses
controlled feature noise and 10% modality dropout with auxiliary prediction,
reliability and uncertainty objectives. Temperature is selected on full-modality
validation predictions only. Test labels do not select checkpoints or temperature.

## Held-out test results

| Method | Accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|
| Training-majority baseline | 49.79% | 0.3324 | 0.5000 |
| Logistic regression with context | 66.23% | 0.6622 | 0.7149 |
| Logistic regression without context | 67.45% | 0.6744 | 0.7299 |
| AdaTrustMSA, final utterance only | 68.51% | 0.6847 | 0.7471 |
| AdaTrustMSA with context | 68.84% | 0.6884 | 0.7517 |

![UR-FUNNY V1 test results](figures/urfunny_round1.png)

The context adapter improves over the context-matched linear baseline by 2.61
percentage points in accuracy. The separately trained final-utterance-only
adapter differs from it by 0.33 percentage points. A single seed cannot establish
a reliable context benefit or statistical significance.

## Missing-modality and feature-noise evaluation

| Available input | Accuracy | Macro-F1 |
|---|---:|---:|
| Text + audio + vision | 68.84% | 0.6884 |
| Text + audio | 68.51% | 0.6851 |
| Text + vision | 67.14% | 0.6714 |
| Audio + vision | 57.54% | 0.5750 |
| Text only | 66.72% | 0.6671 |
| Audio only | 56.72% | 0.5671 |
| Vision only | 53.01% | 0.4948 |

All missing-modality routes mask the same context checkpoint at inference;
they are not separately trained unimodal baselines. Each route uses the
temperature fitted on full-modality validation only. Adding Gaussian noise
with standard deviation 0.5 in standardised feature space gives 68.66% accuracy
and 0.6866 Macro-F1; this is synthetic feature corruption rather than measured
microphone or camera degradation.

## Verification and limitations

The saved prediction CSV reproduces the recorded accuracy, Macro-F1 and
confusion matrix. Saved-checkpoint inference reproduces all 3,290 context
predictions with maximum probability difference 0.0. The original checkpoint
and dataset hashes remain preserved; this documentation update performs no
new training or checkpoint selection.

- One random seed; a first-round feasibility result rather than a final ranking.
- Official V1 rather than cleaned V2. The V2 removal of noisy/overlapping
  instances is not applied. Speaker identities are absent from this V1 package,
  so speaker isolation is not independently verified.
- Sentence means discard within-sentence timing. No raw-video or ASR inference
  and no real application deployment are evaluated.
- The original adaptive modality-acquisition and temporal early-exit policy
  are not evaluated by this adapter.
- Dataset features and trained checkpoints are not redistributed in this
  documentation update. Only aggregate records and numeric predictions are added.

## Recorded evidence

- [Context model metrics and settings](results/urfunny_v1_20261026/context_metrics.json)
- [Final-utterance-only metrics and settings](results/urfunny_v1_20261026/punchline_metrics.json)
- [Data contract and source hashes](results/urfunny_v1_20261026/dataset_manifest.json)
- [All context-model test predictions](results/urfunny_v1_20261026/test_predictions.csv)
- [Checkpoint-reload verification](results/urfunny_v1_20261026/inference_verification.json)
