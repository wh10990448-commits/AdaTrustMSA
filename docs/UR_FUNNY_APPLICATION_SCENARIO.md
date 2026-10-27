# Humor Review Assistance for Speech and Educational Videos

AdaTrustMSA's UR-FUNNY V1 experiment is presented as dataset-backed evidence for
the humor-classification component of a proposed video-review workflow. Its
measured result is **68.84% accuracy and 0.6884 Macro-F1 on 3,290 test samples**
with seed 2026. A complete editing application has not been evaluated.

## Where it could be used

| Practical setting | Intended use | Evidence scope |
|---|---|---|
| Public-talk archives and speech-production teams | Review candidate humorous utterances when preparing excerpts, summaries or talk indexes | Closest to the recorded-public-talk source of UR-FUNNY |
| University lecture-production groups and online-course content teams | Inspect candidate moments with surrounding explanation while preparing optional excerpts | Proposed transfer; ordinary course recordings were not tested |

The [original UR-FUNNY paper](https://aclanthology.org/D19-1211/) constructs
examples from TED talks, with humor labels based on audience-laughter markers
and with preceding context. These benchmark labels are not direct measurements
of teaching quality, learner engagement or an editor's preferred highlights.

## Why the scenario matters

Reviewing a long recording involves finding candidate moments, checking their
meaning in context and deciding what is suitable for the intended audience.
A queue of likely humorous utterances could help an editor focus that review.
Keeping the preceding setup alongside each candidate is important because
an isolated punchline can lose its meaning when removed from the discussion.
This is the proposed operational value; the experiment has not measured how
much review time, if any, such a queue would save.

Humor is a practical test for combining modalities because wording, vocal
delivery and visible expression can provide different cues. The dataset paper
motivates modeling these signals jointly with context. Evaluating this task
therefore tests whether the existing fusion approach can extend beyond
sentiment prediction to another communicative behavior. Missing-modality
checks also provide a controlled way to examine operation when fewer signals
are available; they do not establish field performance with actual recording
faults.

## Proposed workflow

1. Prepare aligned utterance features for text, audio and vision. The current
   experiment uses the supplied V1 features; extraction from new raw recordings
   remains integration work.
2. Include up to eight preceding context sentences and predict whether the
   final utterance is humorous, together with its predicted probability.
3. Present candidates for human review. Source-video playback and time-based
   navigation would require source recordings and verified timestamp mappings,
   which are not supplied by this experimental interface.
4. Let an editor inspect the setup and decide what to retain. Ranking rules,
   thresholds and clip boundaries would need validation for the target workflow.

## What has been measured

| Measurement | Recorded result |
|---|---|
| Dataset and task | Official UR-FUNNY V1; binary humor classification |
| Split | 10,598 training / 2,626 validation / 3,290 test samples |
| Text + audio + vision with context | **68.84% accuracy / 0.6884 Macro-F1** |
| Context-matched linear baseline | 66.23% accuracy / 0.6622 Macro-F1 |
| Separately trained final-utterance-only adapter | 68.51% accuracy / 0.6847 Macro-F1 |
| Same context checkpoint, text + audio route | 68.51% accuracy / 0.6851 Macro-F1 |
| Same context checkpoint, text-only route | 66.72% accuracy / 0.6671 Macro-F1 |
| Repetition | One seed, 2026; a first-round feasibility experiment |

The missing-modality rows reuse the same trained checkpoint and do not
represent independently trained unimodal baselines. The original adaptive
acquisition and temporal early-exit policy is not evaluated here. See the
[experiment report](UR_FUNNY_V1_RESULTS.md) and [recorded metrics](results/urfunny_v1_20261026/context_metrics.json).

## What practical deployment still requires

The next application evaluation should use representative target recordings
with verified utterance boundaries, context and humor annotations, separated
by speaker or recording where possible. It should measure false positives,
missed candidates, precision and recall at a validation-chosen operating point,
and actual review time with and without model assistance. Any claim about
course engagement or learning outcomes would require a separate learner study.

The current evidence is based on pre-extracted features and one random seed.
V1 has not received V2's cleanup of noisy or overlapping instances, and speaker
isolation is not independently verified from the acquired V1 package. These
limits apply when interpreting its use as application evidence.
