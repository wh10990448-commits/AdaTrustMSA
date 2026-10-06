from __future__ import annotations

import numpy as np

from adatrustmsa_metrics import metrics


def test_mmsa_metrics_follow_raw_regression_protocol():
    truth = np.asarray([-2.0, -0.2, 0.0, 0.6, 2.1], dtype=np.float32)
    prediction = np.asarray([-1.6, 0.1, -0.1, 0.7, 1.4], dtype=np.float32)
    labels = (np.sign(truth).astype(np.int64) + 1)
    logits = np.eye(3, dtype=np.float32)[labels] * 4.0

    result = metrics(labels, truth, logits, prediction)

    assert result["mmsa_acc_7"] == 0.8
    assert result["mmsa_non0_acc_2"] == 0.75
    assert result["mmsa_has0_acc_2"] == 0.6
    assert np.isclose(result["mmsa_mae"], 0.32, atol=1e-6)
    assert result["mmsa_corr"] > 0.9
