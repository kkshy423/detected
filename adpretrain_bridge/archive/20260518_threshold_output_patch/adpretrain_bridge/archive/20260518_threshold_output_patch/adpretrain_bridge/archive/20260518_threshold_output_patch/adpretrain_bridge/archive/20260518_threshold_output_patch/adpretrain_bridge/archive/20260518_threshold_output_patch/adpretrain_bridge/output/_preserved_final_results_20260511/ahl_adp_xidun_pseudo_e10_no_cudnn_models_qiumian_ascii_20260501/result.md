# Result Status

更新时间：2026-05-01

## 状态

当前有效作业：`202568.Ghead`，job name `ahl_adp_ascii_qm`，正在运行。

本轮已经完成的关键验证：

- ASCII alias 数据根可用：`cache_alias/models_qiumianxiepai` 指向真实 `models__球面斜拍` 缓存目录。
- 缓存检查通过：train 560、test 339，`feature=(512,14,14)`，`feature_scale=(512,7,7)`。
- AHL 已进入训练，epoch 0 已产生完整 `result.txt` block，没有再出现中文路径错误。
- 关闭 cuDNN wrapper 已生效进入 AHL 主流程；截至当前未复现上一轮 epoch 5 之前的 cuDNN LSTM backward 错误。

## 当前临时指标

这是作业运行中基于当前 `result.txt` 的 train-free 评估，当前只包含 epoch 0 的最后输出，不代表最终 30 epoch 结果。

| Class | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| models_qiumianxiepai | 0.6565 | 0.6557 | 0.8466 | 0.7390 | 0.7123 | 0.7661 |

临时指标文件：`/ghome/huangjd/code/detected/adpretrain_bridge/output/ahl_adp_xidun_pseudo_e10_no_cudnn_models_qiumian_ascii_20260501/metrics_current/ahl_aligned_metrics_full.md`

## 等待完成后补充

- `metrics/ahl_aligned_metrics_full.md`：作业脚本最终评估输出。
- 30 epoch 是否完整跑完。
- 与原始 AHL(DRA) baseline、官方 ADPretrain direct、上一轮西顿伪异常 partial 的对比结论。
