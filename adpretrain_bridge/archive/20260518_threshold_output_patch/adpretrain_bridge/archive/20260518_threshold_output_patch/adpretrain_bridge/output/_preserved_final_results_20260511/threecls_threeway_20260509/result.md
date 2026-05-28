# threecls_threeway_20260509

Status: PBS generated and validated; waiting for user confirmation before qsub.

## Record Point

`2026-05-09 ?????????`

Future workload summaries and weekly reports should count from this record point unless the user states otherwise.

## Experiments

| Group | Description | Output |
| --- | --- | --- |
| A | DRA baseline -> AHL | `ahl_dra_baseline/` |
| B | CLIP/ADPretrain plain feature -> AHL | `ahl_clip_plain/` |
| C | CLIP/ADPretrain CHMM feature -> AHL | `ahl_clip_chmm/` |

## Data Roots

- DRA reference/baseline cache: `/gdata1/huangjd/data/xidun_threecls_adpretrain_dra_resnet18/dra_reference_e30_20260509`
- CLIP plain cache: `/gdata1/huangjd/data/xidun_threecls_adpretrain_clip_base/plain_official_effective_20260509`
- CLIP CHMM cache: `/gdata1/huangjd/data/xidun_threecls_adpretrain_clip_base/chmm_official_effective_draref_e30_20260509`

## PBS Assets

- Job A launcher: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_threecls_threeway_A_dra_baseline_20260509.sh`
- Job A PBS: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/threecls_threeway_A_dra_baseline_20260509.pbs`
- Job B launcher: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_threecls_threeway_B_clip_plain_20260509.sh`
- Job B PBS: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/threecls_threeway_B_clip_plain_20260509.pbs`
- Job C launcher: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_threecls_threeway_C_clip_chmm_20260509.sh`
- Job C PBS: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/threecls_threeway_C_clip_chmm_20260509.pbs`

## Submission Order

1. Submit Job A and Job B after confirmation.
2. Submit Job C only after Job A and Job B finish successfully.

## 2026-05-11 三类三实验结果总结

### 实验范围

- 数据集：`/gdata1/huangjd/data/xidun_mvtec_format_threecls`
- 类别：`models`、`qiusai`、`yuanzhu`
- 对照组：
  - A. `DRA baseline -> AHL`
  - B. `CLIP/ADPretrain plain feature -> AHL`
  - C. `CLIP/ADPretrain CHMM feature -> AHL`
- 训练设置：AHL 三组均跑 30 epoch；B/C 使用 CLIP-base official effective projector；C 使用 DRA train/good 的 per-channel moment 作为 reference。

### 三组平均指标

| 组别 | Accuracy | F1 | AUC-ROC | AUC-PR | AHL forward ms | CLIP feature ms | CHMM ms | 估算端到端 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DRA baseline | 0.9105 | 0.9349 | 0.9031 | 0.9760 | 9.13 | - | - | 9.13* |
| CLIP plain | 0.9033 | 0.9342 | 0.8958 | 0.9701 | 10.21 | 21.31 | - | 31.52 |
| CLIP CHMM | 0.9040 | 0.9317 | 0.8725 | 0.9677 | 10.63 | 20.84 | 0.27 | 31.74 |

`*` DRA baseline 的 9.13 ms 是 AHL feature-level forward 时间，不包含 DRA 特征提取。

### 分类别 AUC-ROC / AUC-PR

| 类别 | DRA AUC-ROC | Plain AUC-ROC | CHMM AUC-ROC | DRA AUC-PR | Plain AUC-PR | CHMM AUC-PR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| models | 0.9074 | 0.9099 | 0.9043 | 0.9443 | 0.9292 | 0.9274 |
| qiusai | 0.8415 | 0.8292 | 0.8443 | 0.9864 | 0.9851 | 0.9865 |
| yuanzhu | 0.9603 | 0.9481 | 0.8689 | 0.9974 | 0.9961 | 0.9894 |

### 结论

1. CLIP/ADPretrain plain replacement 已接近 DRA baseline，但没有超过 DRA。平均 AUC-ROC 低 0.0073，平均 AUC-PR 低 0.0059，F1 基本持平。
2. CHMM 在三类平均上没有带来收益，反而拉低 AUC-ROC 和 F1，尤其在 `yuanzhu` 上 AUC-ROC 从 plain 的 0.9481 降到 0.8689。
3. CHMM 只在 `qiusai` 的 AUC-ROC/AUC-PR 上有轻微提升，但不足以抵消 `yuanzhu` 的明显退化。因此 CHMM 不应作为当前默认特征替换策略。
4. 目前最有价值的主线是 `CLIP/ADPretrain plain feature -> AHL`，因为它在不依赖 DRA 分布校准的情况下已经接近 DRA baseline，且工程链路更简单。
5. 低风险替换基座的阶段性结论是：直接替换可行但不足以稳定超越 DRA；下一步应从“分布校准”转向“保留 CLIP 表征优势并改进 AHL 适配”。

### 下一步实验建议

优先做 D. `CLIP plain + 轻量 AHL adapter/head 对齐`：固定 CLIP/ADPretrain cache，不再做 CHMM，新增解耦 adapter 脚本或 AHL 包装层，只在进入 AHL 前做可训练或可控的尺度/归一化/门控适配。

合理性：CHMM 是手工逐通道均值方差对齐，假设 DRA 的通道分布是 AHL 的最佳输入分布；结果显示该假设不稳，尤其会破坏 `yuanzhu` 的排序能力。相比之下，plain CLIP 已接近 DRA，说明 CLIP 特征本身有信息量，问题更可能是 AHL head 对新特征的适配不足，而不是必须强行匹配 DRA 分布。

建议执行顺序：

1. D1：CLIP plain + feature L2/channel norm 对照，只生成新 cache，不改 AHL 主体。
2. D2：CLIP plain + learnable 1x1/channel affine adapter，仅用 train/good 和训练集伪异常，不使用 test。
3. D3：CLIP plain 与 DRA feature late fusion，对比 `score average` 和 `feature concat + small adapter`，验证 CLIP 是否能补充 DRA。
4. 若 D1/D2/D3 任一组超过 DRA baseline，再扩展到更多类别和更严格重复实验；否则暂停基座替换，转向异常分数融合或 AHL head 重设计。
## 2026-05-11 Precision / Recall 主指标补充

本轮三类三实验的 `ahl_aligned_metrics_full.json` 已直接包含 `precision` 和 `recall` 字段，不需要重新提交 PBS，也不需要重新跑 AHL。以下指标来自现有离线评估结果，阈值策略为 `best-f1`。

### 分类别 Precision / Recall

| 组别 | 类别 | Precision | Recall | F1 | AUC-ROC | AUC-PR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DRA baseline | models | 0.9620 | 0.8042 | 0.8761 | 0.9074 | 0.9443 |
| DRA baseline | qiusai | 0.9222 | 1.0000 | 0.9595 | 0.8415 | 0.9864 |
| DRA baseline | yuanzhu | 0.9431 | 0.9963 | 0.9690 | 0.9603 | 0.9974 |
| CLIP plain | models | 0.7991 | 0.9471 | 0.8668 | 0.9099 | 0.9292 |
| CLIP plain | qiusai | 0.9222 | 1.0000 | 0.9595 | 0.8292 | 0.9851 |
| CLIP plain | yuanzhu | 0.9659 | 0.9866 | 0.9761 | 0.9481 | 0.9961 |
| CLIP CHMM | models | 0.8889 | 0.8466 | 0.8672 | 0.9043 | 0.9274 |
| CLIP CHMM | qiusai | 0.9222 | 1.0000 | 0.9595 | 0.8443 | 0.9865 |
| CLIP CHMM | yuanzhu | 0.9387 | 1.0000 | 0.9684 | 0.8689 | 0.9894 |

### 三类平均 Precision / Recall

| 组别 | Precision | Recall | F1 | AUC-ROC | AUC-PR |
| --- | ---: | ---: | ---: | ---: | ---: |
| DRA baseline | 0.9424 | 0.9335 | 0.9349 | 0.9031 | 0.9760 |
| CLIP plain | 0.8957 | 0.9779 | 0.9342 | 0.8958 | 0.9701 |
| CLIP CHMM | 0.9166 | 0.9489 | 0.9317 | 0.8725 | 0.9677 |

### 按 Precision / Recall 重新解读

- 若以 Precision 为主，DRA baseline 最好，CLIP CHMM 次之，CLIP plain 最低。CLIP plain 在 `models` 上 precision 明显偏低，说明误报更多。
- 若以 Recall 为主，CLIP plain 最好，CLIP CHMM 次之，DRA baseline 最低。CLIP plain 更倾向于检出异常，但代价是 precision 下降。
- 若要求 Precision / Recall 均衡，DRA baseline 仍是最稳基准；CLIP plain 更像高召回方案；CHMM 没有形成稳定优势。
- 后续实验应把 Precision / Recall 作为主指标，AUC-ROC / AUC-PR 作为排序能力参考指标。