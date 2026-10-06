"""CPU smoke test; run inside the AdaTrustMSA PyTorch environment."""

from pathlib import Path
import sys

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from adatrustmsa_corruption import INPUT_KEYS, content_mask, self_test as corruption_test
from adatrustmsa_model import AdaTrustMSAModel
from adatrustmsa_policy import anytime_policy


def make_data():
    ids = np.array(
        [
            [101, 110, 111, 112, 113, 102, 0, 0],
            [101, 120, 121, 122, 102, 0, 0, 0],
            [101, 130, 131, 132, 133, 134, 102, 0],
        ],
        dtype=np.int64,
    )
    attention = (ids != 0).astype(np.int64)
    rng = np.random.default_rng(11)
    audio = rng.normal(size=(*ids.shape, 74)).astype(np.float32)
    vision = rng.normal(size=(*ids.shape, 35)).astype(np.float32)
    content = content_mask(ids, attention)
    audio[~content] = 0
    vision[~content] = 0
    audio[1, 2] = 0
    vision[2, 3] = 0
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "token_type_ids": np.zeros_like(ids),
        "audio": audio,
        "vision": vision,
    }


def main():
    corruption_test()
    config = {
        "vocab_size": 256,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "max_position_embeddings": 64,
        "type_vocab_size": 2,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
    }
    model = AdaTrustMSAModel(
        bert_config=config,
        audio_mean=[0.0] * 74,
        audio_std=[1.0] * 74,
        vision_mean=[0.0] * 35,
        vision_std=[1.0] * 35,
        hidden=8,
        dropout=0.0,
        fusion_layers=2,
        prediction_blending=True,
        attentive_pooling=True,
        quality_attention=True,
        quality_attention_strength=1.0,
    ).eval()
    data = make_data()
    tensors = {key: torch.as_tensor(data[key]) for key in INPUT_KEYS}
    with torch.inference_mode():
        output = model(**tensors)
    assert output["logits"].shape == (3, 3)
    assert output["regression"].shape == (3,)
    assert output["reliability"].shape == (3, 3)
    assert output["conflict_matrix"].shape == (3, 3, 3)
    assert output["interaction_tokens"].shape == (3, 3, 16)
    assert output["expert_logits"].shape == (3, 4, 3)
    assert output["expert_weights"].shape == (3, 4)
    assert output["attention_quality"].shape == (3, 3)
    assert output["classification_blend"].shape == (3, 3)
    assert output["regression_blend"].shape == (3,)
    assert torch.isfinite(output["logits"]).all()
    present = output["fractions"].gt(0).any(dim=1)
    torch.testing.assert_close(
        output["modality_weights"][present].sum(dim=1),
        torch.ones(int(present.sum())),
    )
    torch.testing.assert_close(
        output["conflict_matrix"], output["conflict_matrix"].transpose(1, 2)
    )
    torch.testing.assert_close(
        output["expert_weights"][present].sum(dim=1),
        torch.ones(int(present.sum())),
    )
    rows, summary = anytime_policy(
        model,
        data,
        torch.device("cpu"),
        prefixes=(0.5, 1.0),
        modality_threshold=0.0,
        exit_threshold=0.0,
        minimum_exit_prefix=0.5,
        require_temporal_stability=False,
        batch_size=2,
    )
    assert len(rows) == 3 and summary["samples"] == 3
    assert all(row["modalities_used"] == "T" for row in rows)
    print("AdaTrustMSA smoke test: PASS")


if __name__ == "__main__":
    main()
