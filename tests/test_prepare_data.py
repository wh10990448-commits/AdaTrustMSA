"""NumPy-only test for the full CMU-MOSEI converter."""

from pathlib import Path
import sys

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from prepare_data import DATASETS, convert_split, fit_statistics


def make_block(split: str, count: int, audio_dim: int = 74, vision_dim: int = 35):
    length = 8
    text = np.zeros((count, 3, length), np.int64)
    text[:, 0, 0] = 101
    text[:, 0, 1:5] = np.arange(110, 114)
    text[:, 0, 5] = 102
    text[:, 1, :6] = 1
    audio = np.zeros((count, length, audio_dim), np.float32)
    vision = np.zeros((count, length, vision_dim), np.float32)
    audio[:, 1:5] = 1.0
    vision[:, 1:5] = 2.0
    regression = np.linspace(-2, 2, count, dtype=np.float32)
    classification = np.sign(regression).astype(np.int64) + 1
    return {
        "text_bert": text,
        "audio": audio,
        "vision": vision,
        "regression_labels": regression,
        "classification_labels": classification,
        "id": np.asarray([f"{split}_{index}" for index in range(count)]),
    }


def main():
    train = convert_split(make_block("train", 5), "train")
    valid = convert_split(make_block("valid", 3), "valid")
    mosi = convert_split(
        make_block("mosi", 2, audio_dim=5, vision_dim=20),
        "mosi",
        expected_audio_dim=5,
        expected_vision_dim=20,
    )
    assert train["input_ids"].shape == (5, 8)
    assert valid["audio"].shape == (3, 8, 74)
    assert mosi["audio"].shape == (2, 8, 5)
    assert mosi["vision"].shape == (2, 8, 20)
    statistics, audit = fit_statistics(train)
    assert len(statistics["audio_mean"]) == 74
    assert len(statistics["vision_std"]) == 35
    assert audit["audio_normalization"]["fit_split"] == "train"
    assert sum(DATASETS["mosei"]["splits"].values()) == 22856
    assert sum(DATASETS["mosi"]["splits"].values()) == 2199
    print("AdaTrustMSA data converter test: PASS")


if __name__ == "__main__":
    main()
