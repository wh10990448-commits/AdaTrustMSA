"""Small deterministic checks for the CSD alignment code."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_full_mosei_from_csd import build_sample, finite_mean, overlapping_rows


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0
    unk_token = "[UNK]"

    def tokenize(self, word):
        return [word.lower()] if word else []

    def convert_tokens_to_ids(self, pieces):
        return [1000 + index for index, _ in enumerate(pieces)]


intervals = np.asarray([[0.0, 0.5], [0.5, 1.0], [1.0, 1.5]])
assert overlapping_rows(intervals, 0.5, 1.0) == (1, 2)
mean, observed, bad = finite_mean(
    np.asarray([[1.0, np.nan], [3.0, 5.0]], dtype=np.float32), 0, 2, 2
)
assert observed and bad == 1 and np.allclose(mean, [2.0, 5.0])

sample, facts = build_sample(
    video_id="video",
    segment_index=0,
    label_vector=np.asarray([1.0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
    label_interval=np.asarray([0.0, 1.0]),
    word_features=np.asarray([[b"Good"], [b"movie"]]),
    word_intervals=np.asarray([[0.0, 0.4], [0.5, 0.9]]),
    audio_features=np.ones((2, 74), dtype=np.float32),
    audio_intervals=np.asarray([[0.0, 0.4], [0.5, 0.9]]),
    vision_features=np.ones((2, 35), dtype=np.float32) * 2,
    vision_intervals=np.asarray([[0.0, 0.4], [0.5, 0.9]]),
    tokenizer=FakeTokenizer(),
    sequence_length=6,
)
assert sample["input_ids"].tolist() == [101, 1000, 1000, 102, 0, 0]
assert sample["classification_labels"] == 2
assert sample["audio_observed"].tolist() == [0, 1, 1, 0, 0, 0]
assert sample["vision_observed"].tolist() == [0, 1, 1, 0, 0, 0]
assert facts["text_observed"] and facts["audio_observed"] and facts["vision_observed"]
print("full CSD extractor test: PASS")
