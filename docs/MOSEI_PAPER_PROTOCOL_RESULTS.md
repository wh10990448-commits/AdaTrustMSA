# CMU-MOSEI 标准论文口径结果

本页只比较标准 MMSA `aligned_50.pkl` 的 clean test 结果。项目三分类
Macro-F1、ECE 与早退指标使用另一套输出协议，单独列在后文，不能与这里的
去零 Acc-2/F1 混用。

## 可复现设置

- 数据：CMU-MOSEI MMSA aligned，16,326 / 1,871 / 4,659；
- 数据 SHA-256：`45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791`；
- 输入：长度 50，COVAREP 74 维，FACET 35 维；
- 文本编码器：`bert-base-uncased`，768 维、12 层、12 头；
- 参数量：112,372,385；batch size 16；
- 回归基线按 validation MAE 选 checkpoint；Pareto 精修按 validation 的
  Acc-7/Acc-2/F1/MAE/Corr 综合分数选 checkpoint；
- 最终集成权重只在 validation 网格选择，固定为“回归基线 0.45 + Pareto
  精修 0.55”，不使用 test 标签选择权重。

## 最终 clean test

| 指标 | 当前 Pareto 集成 | E-MoE（aligned） | 差值（当前 − E-MoE） |
|---|---:|---:|---:|
| Acc-7 ↑ | 54.046% | 54.10% | -0.054 pp |
| Acc-2 去零 ↑ | 85.250% | 85.30% | -0.050 pp |
| F1 去零 ↑ | 85.225% | 85.30% | -0.075 pp |
| MAE ↓ | **0.5344** | 0.5360 | **优 0.0016** |

当前方法的 Corr 为 **0.7648**；E-MoE 对照表未报告 Corr，因此不制造差值。
四项中 MAE 已超过 E-MoE，其余三项非常接近，但不能表述为“全面超过”。

## 与此前单模型结果对比

| 方法 | Acc-7 ↑ | Acc-2 去零 ↑ | F1 去零 ↑ | MAE ↓ | Corr ↑ |
|---|---:|---:|---:|---:|---:|
| CQPA + BERT-base 单模型 | **54.37%** | 84.45% | 84.43% | 0.5383 | 0.7562 |
| Pareto 精修单模型 | 53.42% | **85.305%** | **85.287%** | 0.5398 | **0.7665** |
| 最终 Pareto 集成（0.45/0.55） | 54.046% | 85.250% | 85.225% | **0.5344** | 0.7648 |
| E-MoE（aligned） | 54.10% | 85.30% | 85.30% | 0.5360 | 未报告 |

集成牺牲了少量 Pareto 精修单模型的二分类结果，换回了 Acc-7 与 MAE，形成
四项更均衡的最终版本。基线机器可读结果见
`mosei_paper_s1111_20260929.json`；最终结果见
`mosei_pareto_ensemble_20260929.json`。

## 项目功能指标（此前结果，另一口径）

以下是完整 CSD 项目协议上的三分类/可信预测指标，不参与上面的 E-MoE
论文表格比较。

| 指标 | 原浅融合 | 默认深融合 | 深融合 + CQPA |
|---|---:|---:|---:|
| Accuracy | **0.6656** | 0.6615 | 0.6632 |
| Macro-F1 | 0.6217 | 0.6258 | **0.6272** |
| Weighted-F1 | 0.6598 | 0.6612 | **0.6628** |
| MAE ↓ | 0.5857 | 0.5772 | **0.5767** |
| Pearson | 0.7036 | 0.7093 | **0.7100** |
| ECE ↓ | 0.0592 | 0.0357 | **0.0352** |
| Brier ↓ | 0.4479 | 0.4438 | **0.4435** |

动态早退 Macro-F1 依次为 0.6011、0.6132、0.6144；动态早退 MAE 依次为
0.6357、0.6078、0.6083。完整实验条件见 `DEEP_FUSION_RESULTS.md`。

## 复现最终集成

先分别得到 `mosei_paper_s1111/best.pt` 和
`mosei_pareto_s1111/best.pt`，再执行：

```bash
PYTHONPATH=src python src/evaluate_pareto_ensemble.py \
  --data-root /data/mosei_mmsa_standard \
  --regression-checkpoint runs/mosei_paper_s1111/best.pt \
  --classification-checkpoint runs/mosei_pareto_s1111/best.pt \
  --output results/mosei_pareto_ensemble_test.json
```

默认只搜索两个 checkpoint 的混合权重，复现实测的 0.55。脚本把完整
validation 网格、选中权重、test 指标和相对 E-MoE 的差值写入 JSON，便于审计。
