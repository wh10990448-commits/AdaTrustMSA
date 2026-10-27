# AdaTrustMSA Results

This file summarizes measured results included with the repository. Two
evaluation protocols are reported and must not be mixed.

## 1. Standard CMU-MOSEI Paper Protocol

- Dataset: MMSA aligned CMU-MOSEI
- Splits: 16,326 train / 1,871 validation / 4,659 test
- Text encoder: BERT-base
- Acc-2 and F1: non-zero protocol
- Ensemble weight: selected on validation only

| Metric | Current Pareto Ensemble | E-MoE | Difference |
|---|---:|---:|---:|
| Acc-7 | 54.046% | 54.10% | -0.054 pp |
| Acc-2 | 85.250% | 85.30% | -0.050 pp |
| F1 | 85.225% | 85.30% | -0.075 pp |
| MAE | **0.5344** | 0.5360 | **0.0016 lower** |

| Version | Acc-7 | Acc-2 | F1 | MAE | Corr |
|---|---:|---:|---:|---:|---:|
| BERT-base baseline | **54.37%** | 84.45% | 84.43% | 0.5383 | 0.7562 |
| Pareto fine-tuned model | 53.42% | **85.305%** | **85.287%** | 0.5398 | **0.7665** |
| Final Pareto ensemble | 54.046% | 85.250% | 85.225% | **0.5344** | 0.7648 |

The final ensemble mixes the regression-oriented baseline with weight 0.45 and
the Pareto fine-tuned model with weight 0.55. E-MoE Corr was not reported in the
comparison table, so no Corr difference is claimed.

## 2. Project-Oriented Robust Prediction Protocol

These metrics use the full CSD project split and three-class project outputs.
They are not directly comparable to the standard non-zero MMSA metrics above.

| Metric | Shallow Fusion | Deep Fusion | Deep Fusion + CQPA |
|---|---:|---:|---:|
| Accuracy | **0.6656** | 0.6615 | 0.6632 |
| Macro-F1 | 0.6217 | 0.6258 | **0.6272** |
| Weighted-F1 | 0.6598 | 0.6612 | **0.6628** |
| MAE | 0.5857 | 0.5772 | **0.5767** |
| Pearson | 0.7036 | 0.7093 | **0.7100** |
| ECE | 0.0592 | 0.0357 | **0.0352** |
| Brier | 0.4479 | 0.4438 | **0.4435** |

### Dynamic Early Exit

| Metric | Shallow Fusion | Deep Fusion | Deep Fusion + CQPA |
|---|---:|---:|---:|
| Accuracy | 0.6381 | 0.6396 | **0.6411** |
| Macro-F1 | 0.6011 | 0.6132 | **0.6144** |
| MAE | 0.6357 | **0.6078** | 0.6083 |
| Early-exit rate | **68.51%** | 54.20% | 54.76% |
| Mean exit prefix | **70.74%** | 76.77% | 76.50% |
| Mean modalities used | **2.043** | 2.082 | 2.070 |

Full experimental conditions and machine-readable results are available in
`docs/MOSEI_PAPER_PROTOCOL_RESULTS.md`, `docs/DEEP_FUSION_RESULTS.md`, and the
JSON files under `docs/`.

## 3. Application Scenario: Humor Review Assistance

**Intended setting:** public-talk archives, university lecture-production teams
and online-course editing workflows. Flagging potentially humorous utterances
with their context could help an editor prioritise review and avoid extracting
a punchline without its setup. This is the proposed practical value; review
efficiency and educational benefits have not been measured.

This separate application experiment uses official V1 folds: **10,598 train /
2,626 validation / 3,290 test** samples. It predicts binary humor labels from
provided text, audio and visual features. These results belong to this project's
feature adapter and use a different task and protocol from CMU-MOSEI above.
The result measures the utterance-classification component of the proposed
workflow; transfer to ordinary course recordings remains to be evaluated.

| Method | Test accuracy | Test Macro-F1 | Test ROC-AUC |
|---|---:|---:|---:|
| Logistic regression with context | 66.23% | 0.6622 | 0.7149 |
| AdaTrustMSA, final utterance only | 68.51% | 0.6847 | 0.7471 |
| **AdaTrustMSA with context** | **68.84%** | **0.6884** | **0.7517** |

The context adapter exceeds its context-matched linear baseline by **2.61
percentage points** in accuracy. Only seed **2026** was run for this experiment;
this does not establish statistical significance or a published-model ranking.
No original MOSEI checkpoint was used. Raw-video inference and the adaptive
acquisition/early-exit policy were not evaluated.

See [the application scenario and its importance](docs/UR_FUNNY_APPLICATION_SCENARIO.md),
[the complete UR-FUNNY V1 report](docs/UR_FUNNY_V1_RESULTS.md),
[original recorded metrics](docs/results/urfunny_v1_20261026/context_metrics.json)
and [all numeric test predictions](docs/results/urfunny_v1_20261026/test_predictions.csv).
