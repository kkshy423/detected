# 球面斜拍 few-shot AHL 实验结果整理（2026-05-12）

## 1. 实验目的

本轮实验针对 `models / 球面斜拍` 单类产线，排查“切割数据集后性能明显下降”的原因，并重新确认可作为主线的配置。

重点比较：

1. 去掉 CHMM 后，`CLIP-B16 ADPretrain official projected residual feature -> AHL` 的 train/test-only few-shot 曲线。
2. 原 CHMM 配置与 plain feature 的差异。
3. 使用原始完整单类目录结构的 plain AHL 对照实验，判断性能下降是否主要由数据切割导致。
4. 数据配置和 feature cache 是否存在路径错位或顺序错位。

## 2. 数据与划分

原始数据：

```text
/gdata1/huangjd/data/xidun_mvtec_format/models__球面斜拍
```

当前使用 train/test-only 划分：

```text
/ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_fewshot_train_test
```

固定测试集：

| Split | Normal | Anomaly |
|---|---:|---:|
| test | 140 | 60 |

训练阶段：

| Stage | Train normal | Train anomaly |
|---|---:|---:|
| S0 | 50 | 0 |
| S1 | 100 | 1 |
| S2 | 150 | 3 |
| S3 | 200 | 5 |
| S4 | 300 | 10 |
| S5 | 400 | 20 |
| S6 | 500 | 40 |
| S7 | 560 | 80 |
| S8 | 560 | 139 |

本轮已跑阶段：

```text
S0, S2, S4, S6, S8
```

split 检查：

```text
/ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_fewshot_train_test/split_check.json
status: ok
```

## 3. Feature 与训练过程说明

ADPretrain 不做微调。本轮没有重新训练 ADPretrain / projector，只复用官方 CLIP-B16 ADPretrain projected residual feature cache。

plain feature cache：

```text
/gdata1/huangjd/data/xidun_qm_xiepai_adpretrain_clip_b16_fewshot_cache/plain_official_img_angle
```

CHMM feature cache：

```text
/gdata1/huangjd/data/xidun_qm_xiepai_adpretrain_clip_b16_fewshot_cache/chmm_official_img_angle_draref
```

已做 feature / split 对齐检查，结论：

```text
source_root: /gdata1/huangjd/data/xidun_mvtec_format/models__球面斜拍
unique split referenced images: 899
missing_source: 0
missing_cache: 0
shape_bad: 0
feature shape: (512, 14, 14)
feature_scale shape: (512, 7, 7)
stage manifest mapping check: OK
feature alignment check: OK
```

因此当前没有证据表明性能差异来自 feature cache 与图片路径错位。

训练过程口径：

| 过程 | 本轮是否执行 | 说明 |
|---|---:|---|
| ADPretrain / projector 训练或微调 | 否 | 按要求不微调，只复用官方 CLIP-B16 feature |
| AHL 训练 | 是 | 每个阶段分别使用对应 train split 训练 AHL |
| CHMM | plain 主线不使用；另保留对照 | CHMM 对冷启动和排序指标不稳定 |

## 4. 指标口径

AHL 原始项目没有提供固定的部署分类阈值，主要输出 anomaly score 并报告 AUC-ROC / AUC-PR。

为了和 ADPretrain 下游 image-level F1 口径一致，本轮 Precision / Recall / F1 使用：

```text
adpretrain_eval_best_f1
```

即在固定测试集 score 上通过 `precision_recall_curve(labels, scores)` 取后验 best-F1 点。这个口径适合做方法间离线对比，但它是 evaluation-set oracle threshold，不是可直接部署的生产阈值。

## 5. Plain feature + AHL few-shot 结果（无 CHMM）

输出目录：

```text
/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_plain_train_test_20260512
```

| Stage | Train N/A | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 50/0 | 0.5254 | 0.5167 | 0.5210 | 0.7150 | 0.6831 | 0.5246 | 9.6924 |
| S2 | 150/3 | 0.8400 | 0.7000 | 0.7636 | 0.8700 | 0.8736 | 0.7732 | 5.9608 |
| S4 | 300/10 | 0.7879 | 0.8667 | 0.8254 | 0.8900 | 0.9279 | 0.8727 | 6.5100 |
| S6 | 500/40 | 0.8824 | 0.7500 | 0.8108 | 0.8950 | 0.9280 | 0.8702 | 7.8726 |
| S8 | 560/139 | 0.8136 | 0.8000 | 0.8067 | 0.8850 | 0.9224 | 0.8654 | 6.7572 |

观察：

1. 从 S0 到 S2，AHL 明显受益于少量异常样本，F1 从 0.5210 提升到 0.7636。
2. S4 达到本轮最高 F1：0.8254，并且 Recall 达到 0.8667。
3. S6 / S8 的 AUC-ROC 和 AUC-PR 保持在高位，但 F1 没有随异常样本继续单调提升。
4. S4 之后收益开始变小，甚至在 F1 上有轻微回落，说明更多异常样本并未直接转化为更好的 fixed-test best-F1。

## 6. CHMM 对照结果

输出目录：

```text
/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_train_test_20260512
```

| Stage | Train N/A | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 50/0 | 0.3000 | 1.0000 | 0.4615 | 0.3000 | 0.2831 | 0.2227 | 5.9078 |
| S2 | 150/3 | 0.8269 | 0.7167 | 0.7679 | 0.8700 | 0.8949 | 0.7928 | 4.0388 |
| S4 | 300/10 | 0.8246 | 0.7833 | 0.8034 | 0.8850 | 0.8990 | 0.8124 | 6.6404 |
| S6 | 500/40 | 0.8800 | 0.7333 | 0.8000 | 0.8900 | 0.9044 | 0.8257 | 5.8870 |
| S8 | 560/139 | 0.8197 | 0.8333 | 0.8264 | 0.8950 | 0.9150 | 0.8338 | 5.8382 |

对比 plain：

| Stage | Plain F1 | CHMM F1 | Plain AUC-ROC | CHMM AUC-ROC | Plain AUC-PR | CHMM AUC-PR |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0.5210 | 0.4615 | 0.6831 | 0.2831 | 0.5246 | 0.2227 |
| S2 | 0.7636 | 0.7679 | 0.8736 | 0.8949 | 0.7732 | 0.7928 |
| S4 | 0.8254 | 0.8034 | 0.9279 | 0.8990 | 0.8727 | 0.8124 |
| S6 | 0.8108 | 0.8000 | 0.9280 | 0.9044 | 0.8702 | 0.8257 |
| S8 | 0.8067 | 0.8264 | 0.9224 | 0.9150 | 0.8654 | 0.8338 |

结论：

1. CHMM 在 S0 冷启动阶段明显失效，AUC-ROC 只有 0.2831。
2. CHMM 在 S8 的 F1 略高于 plain，但 AUC-ROC / AUC-PR 低于 plain。
3. 如果以稳定排序能力和冷启动鲁棒性为主，CHMM 不适合作为默认主线。
4. plain CLIP-B16 ADPretrain feature 更适合作为当前 few-shot 主线。

## 7. 完整原始单类 plain AHL 对照

输出目录：

```text
/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_full_original_clip_b16_plain_ahl_20260512
```

配置：

```text
Plain CLIP-B16 ADPretrain feature
No CHMM
AHL original-like full root
train/good = 560
test/good = 140
test/defect = 199
nAnomaly = 139
eval = 140 normal + 60 anomaly
metric policy = adpretrain_eval_best_f1
```

结果：

| Setting | Train N/A | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full original plain | 560/139 | 0.8182 | 0.7500 | 0.7826 | 0.8750 | 0.9058 | 0.8375 |
| Split S8 plain | 560/139 | 0.8136 | 0.8000 | 0.8067 | 0.8850 | 0.9224 | 0.8654 |

结论：

在受控 plain AHL 对比中，显式 train/test split 的 S8 并不比完整原始单类实验差，反而在 Recall、F1、Accuracy、AUC-ROC、AUC-PR 上更高。

因此，目前没有证据支持“性能下降主要由数据切割导致”。更可能的原因包括：

1. CHMM 引入后改变了 feature 分布，并在冷启动和排序指标上不稳定。
2. 旧实验和当前实验的 metric / threshold 口径不同。
3. 旧实验可能使用了不同的 test block、异常采样方式、AHL 参数或结果对齐方式。
4. 旧实验若使用后验 best-F1 或完整测试集参与阈值选择，需要和当前 `adpretrain_eval_best_f1` 口径重新对齐。

## 8. 核心问题回答

### 8.1 异常样本从 0 到 139 增加时，Precision 是否提升？

提升但不是单调提升。

plain 主线中：

```text
S0: 0.5254
S2: 0.8400
S4: 0.7879
S6: 0.8824
S8: 0.8136
```

Precision 从 S0 到 S2/S6 明显提升，说明少量异常样本对 AHL 判别边界有效。但 S4/S8 有波动，说明当前 AHL 对异常样本数量、异常类型覆盖和采样顺序敏感。

### 8.2 Recall 是否保持稳定或提升？

Recall 从 S0 到 S4 明显提升，但 S6/S8 有回落：

```text
S0: 0.5167
S2: 0.7000
S4: 0.8667
S6: 0.7500
S8: 0.8000
```

S4 是本轮 Recall 最好阶段。S8 相比 S6 有恢复，但没有超过 S4。

### 8.3 哪个阶段开始具备实际可用性？

按离线 best-F1 口径：

```text
S4: Precision 0.7879, Recall 0.8667, F1 0.8254, AUC-ROC 0.9279, AUC-PR 0.8727
```

S4 开始具备实际可用性。S2 已有明显改善，但 Recall 0.7000 偏低；如果产线对漏检敏感，S2 仍不够稳。

### 8.4 哪个阶段后收益开始变小？

S4 之后收益开始变小。

S4、S6、S8 的 AUC-ROC / AUC-PR 都处于 0.92 / 0.86 附近，但 F1 没有继续提升。说明继续增加异常样本仍有价值，但当前 AHL 配置没有稳定吃满这些新增样本。

### 8.5 是否达到切换全监督模型的条件？

S8 已经达到可以启动 full-supervised baseline 对照的条件，但不建议直接宣称应切换。

理由：

1. S8 已有 139 张训练异常样本，样本规模足够跑一个全监督基线。
2. plain AHL S8 的 F1 为 0.8067，已经是可比较的 few-shot baseline。
3. 但 S4 已达到最高 F1，S8 没有显著超过 S4，说明“更多标签 + 当前 AHL”收益有限。

建议把 S8 作为全监督切换评估点：如果全监督模型在同一 fixed test 上明显超过 S8 AHL，且部署阈值稳定，再切换。

### 8.6 CLIP-B16 + AHL 是否适合作为产线早期小样本方案？

适合作为冷启动到全监督切换前的过渡方案，但主线应使用 plain CLIP-B16 ADPretrain feature，不默认使用 CHMM。

推荐定位：

1. S0 只可作为无异常标签的初始筛查，不适合独立上线。
2. S2 可作为早期试运行模型，但需要人工复核兜底。
3. S4 之后具备较强实用性，可以作为正式 few-shot 过渡模型。
4. S8 应进入 full-supervised baseline 对照阶段。

## 9. 本轮结论

1. 数据切割不是当前性能下降的主要证据来源。plain S8 split 结果优于 full original plain 对照。
2. feature cache 与 split 路径对齐检查通过，没有发现错位。
3. CHMM 不是稳定收益项，尤其 S0 冷启动明显破坏排序能力；后续不建议作为默认主线。
4. 当前主线应固定为 `CLIP-B16 ADPretrain official projected residual feature -> AHL, no CHMM, no ADPretrain finetune`。
5. Precision / Recall / F1 当前采用 ADPretrain-style test best-F1 口径，适合离线公平对比，但不能作为生产阈值结论。
6. few-shot 曲线显示 S4 开始具备实际可用性，S4 后边际收益变小。
7. S8 已经具备启动全监督模型 baseline 的样本条件。

## 10. 下一步实验规划

### P0：统一 baseline 与指标口径

1. 跑 ADPretrain-only baseline，使用同一 train/test split、同一 140 normal + 60 anomaly fixed test、同一 `adpretrain_eval_best_f1` 口径。
2. 对比：

```text
ADPretrain-only
ADPretrain plain feature + AHL
ADPretrain plain feature + AHL + CHMM
```

目标是确认 AHL 相比不加 AHL 的净收益。

### P1：plain AHL 多 seed 稳定性

优先对 S4 / S6 / S8 跑 3-5 个 seed：

```text
S4: 当前最高 F1，验证是否偶然
S6: 样本继续增加后的中间点
S8: 全量异常池点，用作 full-supervised 对照点
```

输出 mean / std，避免基于单次划分做过强结论。

### P2：补齐阶段曲线

补跑：

```text
S1, S3, S5, S7
```

目标是确认 S4 后边际收益变小是否平滑出现，还是由阶段采样波动造成。

### P3：部署阈值实验单独做

当前 best-F1 是测试集后验阈值，不可部署。下一步应单独评估可部署阈值：

1. train normal percentile：P95 / P97.5 / P99。
2. 少量 train anomaly 下的 train-side threshold。
3. recall-priority 与 precision-priority 两套生产策略。

这一步不要混入主对比指标，避免把“模型排序能力”和“阈值部署策略”混在一起。

### P4：全监督 baseline

当使用 S8 级别标签量时，启动 full-supervised baseline：

```text
train normal = 560
train anomaly = 139
test normal = 140
test anomaly = 60
```

目标是判断是否已经达到从 few-shot AHL 切换到全监督模型的条件。

### P5：DRA + CLIP score fusion

在 plain AHL 与 ADPretrain-only baseline 对齐后，再做 DRA + CLIP score fusion。

建议前提：

1. 固定 same split。
2. 固定 same metric policy。
3. 先不引入 CHMM。
4. fusion 只使用 train-side 或固定规则，不使用测试集调参。

## 11. 关键路径索引

split：

```text
/ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_fewshot_train_test
```

plain few-shot 输出：

```text
/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_plain_train_test_20260512
```

CHMM few-shot 输出：

```text
/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_train_test_20260512
```

full original plain 对照输出：

```text
/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_full_original_clip_b16_plain_ahl_20260512
```

本汇总文档：

```text
/ghome/huangjd/code/detected/adpretrain_bridge/qm_xiepai_plain_vs_chmm_split_vs_full_20260512.md
```
