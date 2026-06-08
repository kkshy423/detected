# 20260606 Preprocess Equivalence And Paired Benchmark V1

## Scope

- Stage: S8
- Samples: calib 149, test 250, steady test 245 after warmup 5
- Reference: n_ref=5, preload setup not counted into per-image latency
- PBS job: final run `210557.Ghead`, resource `#PBS -l nodes=1:gpus=1:a`, exit_status=0
- Script: `/ghome/huangjd/code/detected/adpretrain_bridge/benchmark_preprocess_equivalence_paired.py`
- Output root: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260606_preprocess_equivalence_and_paired_benchmark_v1`
- Archived invalid/intermediate outputs:
  - `output/20260606_preprocess_equivalence_and_paired_benchmark_v1_prephase_zero_archive_`
  - `output/20260606_preprocess_equivalence_and_paired_benchmark_v1_order_biased_archive_210556`

## Timing Definition

本轮核心生产口径是 `decoded_image_to_threshold_ms`：从同一张已解码 PIL image/array 开始，到阈值判定结束。`image_load_decode_ms` 只记录，不计入当前生产预加载图片口径。

三种口径：

| item | definition | used as production core |
| --- | --- | --- |
| `file_end_to_end_ms` | PIL decode + preprocess + ADP/AHL + threshold | No |
| `decoded_image_to_threshold_ms` | decoded image + preprocess + ADP/AHL + threshold | Yes |
| `tensor_to_threshold_ms` | encoder input tensor + ADP/AHL + threshold | Model-only reference |

GPU tensor 路径的准确命名是：CPU PIL decode + CPU PIL-to-uint8-tensor + GPU tensor resize/crop/normalize。它不是 full GPU decode。

## Latency Summary

| metric | cpu_pil median | cpu_pil mean | cpu_pil P95 | gpu_tensor median | gpu_tensor mean | gpu_tensor P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| file_end_to_end_ms | 91.83 | 102.46 | 137.31 | 49.37 | 72.74 | 123.49 |
| decoded_image_to_threshold_ms | 82.65 | 92.18 | 126.56 | 39.94 | 62.47 | 109.73 |
| tensor_to_threshold_ms | 29.56 | 56.85 | 103.20 | 29.22 | 55.01 | 100.68 |
| image_load_decode_ms | 7.63 | 10.27 | 12.33 | 7.63 | 10.27 | 12.33 |
| preprocess_total_ms | 25.73 | 35.33 | 60.71 | 5.28 | 7.46 | 20.60 |

关键观察：

- 图片加载/解码 median 为 7.63 ms，但本轮按产线预加载假设不计入核心口径。
- CPU-PIL 预处理 median 为 25.73 ms，P95 为 60.71 ms。上一轮 55.3 ms 不是稳定的纯预处理 median，更像混入了 decode、顺序偏差或尾部慢样本。
- GPU tensor 预处理 median 降到 5.28 ms，P95 为 20.60 ms。有效降低的是 decoded image 到 encoder input tensor 这段。
- `tensor_to_threshold_ms` 在交替顺序后两种 backend median 基本一致，约 29 ms，说明模型本体计时不应被解释成 CPU/GPU 预处理差异。
- 模型段仍存在首轮/周期性慢批次：按 score position 看，第一计分位置 median 约 93-95 ms，第二计分位置 median 约 28.6 ms。这是模型侧/调度侧尾延迟，和预处理迁移不是同一个问题。

## Preprocess Equivalence

| item | mean | median | P95 | max |
| --- | ---: | ---: | ---: | ---: |
| tensor max_abs_diff | 0.26429849 | 0.25923347 | 0.34014845 | 0.44980216 |
| tensor mean_abs_diff | 0.00325363 | 0.00326620 | 0.00343061 | 0.00361134 |
| tensor p95_abs_diff | 0.00934833 | 0.00935064 | 0.00950343 | 0.00968133 |
| tensor p99_abs_diff | 0.04108460 | 0.04115092 | 0.04678983 | 0.05227986 |
| score_abs_diff | 0.01657127 | 0.01314229 | 0.04390411 | 0.12289089 |

固定 CPU calib threshold `0.575703978539` 下：

- prediction changed all: 11
- prediction changed test: 7

结论：GPU tensor 预处理没有达到严格语义等价。虽然平均 tensor diff 不大，但局部像素差和 score diff 已经足以改变阈值附近样本的预测。

## Metrics

### Fixed CPU Threshold

| backend | threshold | Acc | P | R | F1 | AUC-ROC | AUC-PR | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu_pil | 0.575704 | 0.9160 | 0.9153 | 0.7714 | 0.8372 | 0.9612 | 0.9167 | 54 | 5 | 175 | 16 |
| gpu_tensor | 0.575704 | 0.9120 | 0.9615 | 0.7143 | 0.8197 | 0.9630 | 0.9097 | 50 | 2 | 178 | 20 |

### Backend Calibrated Threshold

| backend | threshold | Acc | P | R | F1 | AUC-ROC | AUC-PR | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu_pil | 0.575704 | 0.9160 | 0.9153 | 0.7714 | 0.8372 | 0.9612 | 0.9167 | 54 | 5 | 175 | 16 |
| gpu_tensor | 0.582367 | 0.9040 | 0.9600 | 0.6857 | 0.8000 | 0.9630 | 0.9097 | 48 | 2 | 178 | 22 |

解读：

- GPU tensor 的 AUC-ROC 基本不差，说明整体排序能力没有明显崩坏。
- 但在当前固定阈值策略下，GPU tensor 路径 Recall 从 0.7714 降到 0.7143，F1 从 0.8372 降到 0.8197。
- 即使按 backend 重新用 calib 选阈值，GPU tensor F1 仍降到 0.8000，Recall 降到 0.6857，说明它不是简单阈值平移问题。

## Conclusion

1. 按产线预加载假设，`image_load_decode_ms` 应忽略。本轮主看 `decoded_image_to_threshold_ms`。
2. GPU tensor 预处理在速度上有效：预处理 median 从 25.73 ms 降到 5.28 ms，decoded-image-to-threshold median 从 82.65 ms 降到 39.94 ms。
3. 但 GPU tensor 预处理当前不能直接替代 CPU-PIL 生产口径，因为它与 CPU-PIL baseline 不严格等价，并造成 7 个 test 样本预测变化。
4. 当前应该把 `cpu_pil` 继续作为指标主口径，把 `gpu_tensor` 作为加速候选路径。下一步若继续推进，应优先解决预处理等价性，而不是继续优化模型本体。
5. 模型本体 `tensor_to_threshold_ms` 的 backend median 基本一致，AHL/ADP 本体没有因为预处理迁移发生本质加速；真正被优化的是 decoded image 到 encoder tensor 的预处理路径。
