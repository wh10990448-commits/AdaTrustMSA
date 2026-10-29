# Application experiments: 2026-10-29 evidence update

These are measurements from this project. The complete test predictions and
metric-checking script are included; raw datasets, pretrained weights and large
training caches are not. Earlier benchmark tables and the original UR-FUNNY V1
68.84% result are preserved as historical records.

## Primary reported results

| Dataset | Original model | Validation-selected extension | Measurement unit |
|---|---|---|---|
| CMU-MOSEI | Nonzero Acc2 85.25%; weighted F1 0.8523; MAE 0.5344; Corr 0.7648 | **86.36%; 0.8636; 0.5133; 0.7905** | 3 new branch seeds sharing the original seed 1111 frozen two-model ensemble; 4,659 test rows each; Acc2/F1 use 3,634 nonzero targets |
| MELD sentiment | Accuracy 66.28%; Macro-F1 0.6168 | **68.07%; 0.6466** | 3 new branch seeds sharing the original seed 2026 frozen text/audio adapter; 2,610 test utterances each |
| MUStARD | Accuracy 71.11%; Macro-F1 0.7110 | **71.50%; 0.7149** | 3 original backbone seeds, each with 5 official outer folds; all 690 examples scored once per seed |
| CH-SIMS v2(s) | Acc2 81.59%; MAE 0.3014 | **81.56%; 0.2982** | 3 original backbone seeds; all 1,034 tests each; native[-1,1] regression |
| MIntRec | Protected full-route anchor 74.16%; Macro-F1 0.7119 | **74.83%; 0.7219** | Prespecified validation-selected branch seed 2026 on original backbone 2026; 445 tests; 20 intents |
| UR-FUNNY V1, preserved original | Context-free adapter 68.51%; Macro-F1 0.6847 | **Context adapter 68.84%; 0.6884** | Historical seed 2026 run; 3,290 tests; no new BGE result |

The units differ: three branch seeds on one frozen original backbone/pair are
not three independent full-backbone repetitions. The MIntRec row is its selected
single primary run, not its three-branch mean or its best test seed. MOSEI/MELD
ran on RTX 4090 with PyTorch 2.0.1+cu117. MUStARD/CH-SIMS and the new MIntRec BGE
branch ran on CPU with PyTorch 2.6.0+cpu. Device changes do not account for a
claimed direct CPU/GPU improvement on the same model; these are separate task
extensions with additional learned parameters.

## Candidate scores and validation protection

The primary table uses the separately declared validation safeguard when the
experiment provides one. The candidate and fallback are retained in the
published prediction files:

| Task | Positive-strength candidate | Validation safeguard |
|---|---|---|
| MUStARD | 71.45% accuracy; 0.7144 Macro-F1 | 71.50%; 0.7149; 13/15 fold models use the new branch,2 retain the prior SAM anchor |
| CH-SIMS | 81.75% Acc2; 0.2956 MAE | 81.56%; 0.2982; 2/3 models use the new branch,1 retains the prior SAM anchor |
| MOSEI | 86.36% nonzero Acc2; 0.5133 MAE | Same; all 3 runs select the new branch |
| MELD | 68.07% accuracy; 0.6466 Macro-F1 | Same; all 3 runs select the new branch |

CH-SIMS's better candidate test score is not substituted for the validation
choice. MIntRec's other branch seeds and R-Drop trial are not promoted by their
test ranking. Old original checkpoints and inference APIs remain preserved.

## Which baselines were actually exceeded?

| Task | Executed comparison | Conclusion |
|---|---|---|
| MIntRec | TCL-MAP author-code rerun seed 2026: 74.61%, Macro-F1 0.7142; selected extension 74.83%,0.7219 | One net additional correct prediction on 445 tests. Extra BGE capacity/compute; the rerun differs from the paper's complete setup. No stable strong-baseline/SOTA claim. |
| MOSEI | Original fixed ensemble 85.25%/MAE 0.5344 versus 86.36%/0.5133 | Improves its own original model; no recent strong-model rerun is established. |
| MELD | Original TA adapter 66.28%/0.6168; logistic TA 50.38%/0.4242; extension 68.07%/0.6466 | Exceeds own old model and the traditional logistic baseline; no recent strong-model rerun is established. |
| MUStARD | Traditional SVM 71.59%/0.7157 versus safeguard 71.50%/0.7149 | Still below SVM. |
| CH-SIMS | Ridge 82.11% Acc2, MAE 0.3123, Corr 0.7224 versus safeguard 81.56%,0.2982,0.7109 | Improves MAE and Acc3/Acc5; does not beat Ridge Acc2, binary weighted F1 or Corr. |
| UR-FUNNY V1 | Context logistic 66.23% versus preserved context adapter 68.84% | Historical result exceeds a traditional baseline; new BGE and recent strong-model comparisons are unavailable. |

SVM, Ridge and logistic regression are traditional controls. The TCL-MAP row is
an actual paper-model rerun, not a claim to reproduce its published score or a
claim that it is a current top-three model. No dataset is claimed to have
established stable superiority over recent strong baselines.

## Method variants and training settings

The conceptual extension uses CLS evidence, gradual feature noise, an
orthogonal local-text residual, a SAM parameter mean, route protection, and a
separate BGE text expert with LoRA. Their concrete implementations differ:

- **MIntRec:** the original selected CLS/curriculum checkpoint, local residual
  and SAM head are frozen. A pinned English BGE expert adapts its final 2 blocks
  with rank 8 Q/V/FFN updates and a 20-class head. The old text-only route remains
  exactly 73.48%. The selected branch trained 10 epochs and chose epoch 5,
  alpha 1/temperature 1. It adds 33,360,000 pretrained parameters and 93,716 trained
  parameters; it has not demonstrated balanced audio/vision contributions.
- **MOSEI:** reuse the original 0.45/0.55 BERT ensemble and preceding trained
  orthogonal/SAM regression head. English BGE receives the same 50 token IDs
  after canonical vocabulary verification. Its new scalar head predicts
  3*tanh(head(normalizedCLS)), with train-only Ridge initialization on target/3
  and SmoothL 1 beta 0.5. Validation MAE chooses convex mixing; the safeguard
  checks all five standard metrics. The original text/zero-AV route is retained.
- **MELD:** retain the original frozen text/audio feature adapter and train a
  local-evidence head on frozen BGE CLS/word states, with gradual feature noise
  and 3 SAM refinements whose selected parameters are averaged. An independent
 3-class BGE/LoRA expert then adds centered logits. The original text-only
  route bypasses these additions. No vision feature is invented.
- **MUStARD/CH-SIMS:** retain the earlier feature-adapter/SAM anchors and add
  BGE/LoRA. The main backbone is not rebuilt as MIntRec's fine-tuned BERT CLS
  model. English BGE uses 64 context+target positions on MUStARD; Chinese BGE
  uses 50 positions and a[-1,1] tanh regressor with SmoothL 1 beta 0.1 on CH-SIMS.
  Regression uses convex mixing; classification uses centered logit addition.
- **UR-FUNNY V1:** the preserved 68.84% run uses supplied GloVe/COVAREP/OpenFace
  features, up to 8 context sentences and the earlier fusion adapter. It is not
  a BGE run, not cleaned V 2 and not a raw-video inference result.

BGE branch settings are AdamW, LoRA LR 0.0002, head LR 0.001, weight decay 0.01,
batch 32, gradient clip 1,10% warmup/linear decay, maximum 16 epochs, minimum 4,
patience 5, rank 8 and LoRA dropout 0.05. Classification uses CE smoothing 0.05 and
a train-only C=10 logistic warm head. Fusion strengths/temperatures and epochs
are selected on validation; no test-best seed is selected. The existing SAM
averages local-head parameters; it does not average independently initialized
LoRA A/B factors or average test scores into a deployed model.

## Protocol and integrity notes

MOSEI retains 16,326/1,871/4,659 rows and its standard nonzero binary protocol.
MELD retains 9,989/1,109/2,610 utterances and 3 sentiment labels, not 7 emotion
classes. MUStARD uses 414/138 inner selection and 552/138 outer refit/test, with
the official speaker-dependent folds; inner backbones exclude validation rows.
CH-SIMS uses 2,722/647/1,034 rows from the acquired 4,403-clip feature release,
although the official project describes 4,402. It uses feature-package regression
labels, which differ from the separate metadata CSV for some clips; metadata
labels are not substituted. CH-SIMS Acc2 includes zeros as nonpositive,
unlike MOSEI's nonzero Acc2.

Complete reloaded-state/input checks passed for every reported new model pair,
and the original MOSEI and MELD predictions were reproduced exactly. The public
CSV/JSON evidence supports metric recomputation; these test files alone do not
prove training isolation or replace full training logs and checkpoints.

MELD's joint source package physically includes labels for all splits, and its
setup constructed test target arrays before training. Only train/validation
targets enter fitting, calibration and model/strength selection. An earlier
metadata flag saying test targets were not loaded was inaccurate and is not
repeated as a physical-loading claim here. No test-label training or selection
is established by the mere presence of labels in that joint package.

Official test sets have been viewed in prior development rounds. These are
exploratory, adaptively developed experiments, not an untouched confirmation,
statistical significance result or deployment validation. Extra text-encoder
capacity/compute is part of the new setting. Reported improvements do not
measure review-time savings, user satisfaction or true customer-service use.

## Other experiments and missing inputs

ICT-MMMO's historical three-seed adapter reached 71.67% on 80 tests, below its
77.50% linear baseline. Its 340-row release lacks full transcripts and original
IDs for matching new BGE inputs. UR-FUNNY V1 likewise lacks the needed matched
raw transcripts/vocabulary for this addition. No new BGE score is reported for
either, and UR-FUNNY V2 is not substituted. Simulated-flight EEG/fNIRS and
DAiSEE are not newly validated by this update; emotion/intent benchmark results
are not evidence of aviation fatigue detection.

## Checking the published numbers

```bash
python scripts/verify_application_evidence.py
```

Prediction-file SHA-256 checks normalize CRLF to LF so Windows and Linux
checkouts verify the same evidence. Original source-artifact hashes retain
their raw-byte convention. The command verifies every included prediction-file SHA, per-seed sample count,
within-test ID uniqueness and the aggregate metrics in the manifest. It uses
NumPy and scikit-learn. UR-FUNNY's original published result is also
checked without changing its files. Numeric evidence contains no raw transcripts,
video, private machine paths, credentials or pretrained/model checkpoint files.
