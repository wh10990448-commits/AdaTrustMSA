# Dataset-backed application scenarios

The project evaluates predictor components that could support the following
human-review workflows. A dataset score establishes performance on the stated
benchmark task; it does not establish deployment, productivity or user outcomes.

| Proposed setting | Input and output | Why it matters | Measured evidence | What still needs validation |
|---|---|---|---|---|
| Product-research teams and spoken-opinion video review | Aligned text/audio/vision to continuous sentiment and polarity | Organise opinions at segment level and retain material for qualitative review | MOSEI: 86.36% nonzero Acc2,0.8636 nonzero weighted F1,0.5133 MAE | Real product interviews, retrieval relevance, review efficiency and user outcomes |
| Conversation sentiment review and support-research tools | Text/audio utterances to negative/neutral/positive labels | Surface potentially relevant turns while keeping a person's review in the loop | MELD: 68.07% accuracy,0.6466 Macro-F1 | Customer-call recordings, dialogue-level context, service outcomes; no vision tested |
| Chinese media-asset tagging and clip review | Chinese text/audio/vision to native[-1,1] sentiment | Add polarity/intensity metadata alongside ordinary keyword indexing | CH-SIMS safeguard: 81.56% Acc2,0.2982 MAE | Real archive data, search quality and time saved |
| Sarcasm hints for annotation and dialogue review | Target and context text/vision plus target audio to sarcasm probability | Avoid interpreting an ambiguous line from literal wording alone; preserve context for human checking | MUStARD safeguard: 71.50% accuracy,0.7149 Macro-F1 | Unseen speakers, non-TV conversations and reviewer benefit |
| Video-assistant and support-intent triage prototypes | Text/audio/vision to 20 intent labels | Distinguish the action a speaker requests from their emotional tone | MIntRec selected seed 2026: 74.83% accuracy,0.7219 Macro-F1 | Real service traffic, operational routing, uncertainty thresholds and balanced modality contribution |
| Public-talk and educational-video review | Supplied context+target T/A/V features to humor label | Find candidate moments without losing the preceding setup when making excerpts | Preserved UR-FUNNY V1: 68.84% accuracy,0.6884 Macro-F1 | Actual course recordings, preferred highlights, clip boundaries and editing efficiency |

## Proposed review workflow

1. Segment a recording and prepare aligned, permitted text/audio/visual inputs.
   The current experiments often use supplied features; a new raw-media
   extraction/ASR pipeline is integration work, not a benchmark result here.
2. Run the task-specific predictor and retain the sample ID, task label/score and
   available context. Sentiment, sarcasm, humor and intent are different targets.
3. Let a reviewer filter or sort candidate clips and inspect the recording.
   Confidence is not treated as a calibrated probability unless calibrated and
   validated for that task and operating domain.
4. Collect real-domain labels and measure both prediction quality and the
   intended workflow outcome before making deployment or efficiency claims.

## Evidence scope and importance

MOSEI motivates segment-level analysis of expressed opinions; it is not a test
of a particular company's product interviews. MELD, MUStARD and MIntRec are
scripted-dialogue benchmarks and serve as research proxies for conversation
review. Their scores do not demonstrate real customer-service performance.
CH-SIMS evaluates Chinese sentiment tagging, not a finished retrieval engine.
UR-FUNNY V1 evaluates humor in recorded public talks; ordinary courses were not
tested. The practical value described above is a proposed use of these
predictors, not an observed reduction in staff workload or measured commercial
benefit.

The experiments add pretrained BGE capacity where matched transcripts/token
IDs exist. They preserve old checkpoints and use validation-selected settings.
Different datasets retain different original anchors and task heads, so the
same conceptual additions do not imply identical full architectures or equal
compute. No adaptation result is used as proof of aviation-fatigue monitoring.

## Results and reproducible numeric evidence

See [the dated experiment report](APPLICATION_EXPERIMENTS_20261029.md),
[metrics JSON](results/application_scenarios_20261029/metrics.json) and
[per-file provenance](results/application_scenarios_20261029/provenance.json).
All published numeric test predictions can be checked with:

```bash
python scripts/verify_application_evidence.py
```

This command verifies hashes, sample coverage and reported metrics. It does not
retrain models, recreate raw media or independently establish training/test
isolation from test files alone. Checkpoints, datasets and large caches are not
distributed in this documentation update. The original UR-FUNNY result and its
[detailed scenario](UR_FUNNY_APPLICATION_SCENARIO.md) remain intact.
