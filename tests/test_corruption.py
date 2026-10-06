"""NumPy-only tests for noise, occlusion, prefixes and cascade dropout."""

from pathlib import Path
import sys

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from adatrustmsa_corruption import content_mask, corrupt_batch


def make_data() -> dict:
    ids = np.array([[101, 111, 112, 113, 114, 102, 0]], dtype=np.int64)
    attention = (ids != 0).astype(np.int64)
    content = content_mask(ids, attention)
    audio = np.zeros((*ids.shape, 74), np.float32)
    vision = np.zeros((*ids.shape, 35), np.float32)
    audio[content] = 1.0
    vision[content] = 1.0
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "token_type_ids": np.zeros_like(ids),
        "audio": audio,
        "vision": vision,
    }


def main() -> None:
    clean = make_data()
    noisy, _, audit = corrupt_batch(
        clean,
        np.random.default_rng(1),
        probability=1.0,
        ratio_range=(0.5, 0.5),
        gaussian_probability=1.0,
        modality_subsets=("AV",),
    )
    assert audit[0]["kind"] == "gaussian"
    assert not np.array_equal(noisy["audio"], clean["audio"])
    assert not np.array_equal(noisy["vision"], clean["vision"])

    occluded, _, audit = corrupt_batch(
        clean,
        np.random.default_rng(2),
        probability=1.0,
        ratio_range=(0.5, 0.5),
        gaussian_probability=0.0,
        modality_subsets=("AV",),
    )
    assert audit[0]["kind"] == "missing"
    assert np.count_nonzero(occluded["audio"]) < np.count_nonzero(clean["audio"])

    text_only, reliability, audit = corrupt_batch(
        clean,
        np.random.default_rng(3),
        probability=0.0,
        cascade_dropout_probability=1.0,
    )
    assert audit[0]["cascade_route"] in {"T", "TA"}
    assert not text_only["vision"].any()
    assert reliability[0, 2] == 0.0
    print("AdaTrustMSA corruption test: PASS")


if __name__ == "__main__":
    main()
