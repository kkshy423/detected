# 20260608 encoder latency breakdown + S8 alpha probe v1 - 最终诊断报告

完成时间: 2026-06-09
项目: `/ghome/huangjd/code/detected/adpretrain_bridge`
实验性质: 诊断性、decision-value；不改算法、不改主线、不改 split。

## 1. PBS 与产物

| 实验 | job id | 状态 | 产物 |
|---|---|---|---|
| A encoder latency breakdown | `211466.Ghead` | 完成 | `encoder_latency_breakdown.md/.csv`, `encoder_latency_backbone_meta.json` |
| B S8 alpha probe | `211467.Ghead` | 完成 | `s8_alpha_probe_diagnostic_not_main_table_update.md/.csv`, `s8_alpha_probe_gap_diagnostic_not_main_table_update.csv` |

stderr 只有 xFormers/FutureWarning 类 warning，没有 error/traceback/OOM。

## 2. 实验 A: encoder 慢根因诊断

设置:

- 同一 encoder: `dinov2-large`
- 同一输入: CPU/PIL backend 预先准备好的 GPU tensor
- batch=1, input `[1,3,224,224]`
- N=160 test images, warmup=20, seed=0
- 输出 feature shape: `[1,1024,16,16] × 4`
- 参数量: 304.369M
- ADPretrain `models/dino.py` 映射确认: `dinov2-large` -> DINOv2 ViT-L/14

结果:

| TF32 | 计时方式 | encoder ms | 说明 |
|---|---|---:|---|
| OFF | synced | 18.2257 | 当前等价性口径，逐 forward 前后 sync |
| OFF | pure | 18.1474 | 全部 forward 后 sync 一次，N 张均摊 |
| ON | synced | 11.2931 | 历史默认/乐观口径候选 |
| ON | pure | 11.1467 | TF32 ON 的吞吐均摊 |

归因:

- TF32 是主要开关: OFF -> ON 速度提升 `1.61x`，约省 `6.93 ms`。
- 同步开销不是主因: OFF 下 synced-pure 只差 `0.078 ms`，ON 下只差 `0.146 ms`。
- backbone 本身是大模型: DINOv2 ViT-L/14, 304M 参数，batch=1 @224 下 encoder-only 地板约 `11-18 ms`。本轮没有复现 “6 ms encoder”。

对时间叙事的含义:

- “6ms encoder”应视为历史乐观口径，可能来自 TF32、更小 backbone、更高 batch 均摊或不同计时上下文。
- 当前等价性口径 TF32 OFF 下 encoder-only 约 18ms；如果再加 projector/reference/AHL 等共同模型段，端到端模型地板仍是主要部分。
- 本实验不决定生产是否打开 TF32；若要开 TF32，必须另做数值等价性评估。

## 3. 实验 B: S8 alpha=0.70 可疑性验证

硬边界:

- 本实验只诊断方向，**不是主表更新**。
- 当前在线 AHL/bridge 口径与旧 v2 main table 口径尚未对齐。
- 即使 alpha=0.35 更好，也不能据此修改 bridge v2 主表。主表 alpha 变更前置条件是 AHL score 口径对齐。

设置:

- Stage: S8 only
- Backend: `cpu_pil`, `gpu_tensor_uint8_aa_true`
- Alpha: `0.70`, `0.50`, `0.35`
- 阈值: S8 best-F1，calib 选阈值，test 只报告
- seed=0, batch=1

结果:

| alpha | backend | threshold | P | R | F1 | Acc | AUC-PR | TP | FP | TN | FN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | cpu_pil | 2.6908 | 0.8806 | 0.8429 | 0.8613 | 0.9240 | 0.9275 | 59 | 8 | 172 | 11 |
| 0.70 | gpu_tensor_uint8_aa_true | 3.0940 | 0.8545 | 0.6714 | 0.7520 | 0.8760 | 0.9178 | 47 | 8 | 172 | 23 |
| 0.50 | cpu_pil | 2.8303 | 0.8833 | 0.7571 | 0.8154 | 0.9040 | 0.9314 | 53 | 7 | 173 | 17 |
| 0.50 | gpu_tensor_uint8_aa_true | 2.5812 | 0.8793 | 0.7286 | 0.7969 | 0.8960 | 0.9234 | 51 | 7 | 173 | 19 |
| 0.35 | cpu_pil | 1.8046 | 0.8824 | 0.8571 | 0.8696 | 0.9280 | 0.9289 | 60 | 8 | 172 | 10 |
| 0.35 | gpu_tensor_uint8_aa_true | 1.6590 | 0.8806 | 0.8429 | 0.8613 | 0.9240 | 0.9220 | 59 | 8 | 172 | 11 |

CPU/GPU gap:

| alpha | CPU F1 | GPU F1 | dF1 | CPU R | GPU R | dR | CPU AUPR | GPU AUPR | dAUPR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 0.8613 | 0.7520 | -0.1093 | 0.8429 | 0.6714 | -0.1714 | 0.9275 | 0.9178 | -0.0097 |
| 0.50 | 0.8154 | 0.7969 | -0.0185 | 0.7571 | 0.7286 | -0.0286 | 0.9314 | 0.9234 | -0.0079 |
| 0.35 | 0.8696 | 0.8613 | -0.0083 | 0.8571 | 0.8429 | -0.0143 | 0.9289 | 0.9220 | -0.0070 |

预注册检查:

- P1: CPU 侧 S8 alpha=0.35 的 F1 >= alpha=0.70。**PASS**: 0.8696 vs 0.8613。
- P2: GPU-CPU 的 S8 F1/Recall gap 在 alpha=0.35 下小于 alpha=0.70。**PASS**: dF1 从 -0.1093 收窄到 -0.0083，dR 从 -0.1714 收窄到 -0.0143。

诊断含义:

- “S8 alpha=0.70 可疑”这个方向得到支持。
- alpha=0.35 在该在线诊断口径下同时提高 CPU F1、显著收窄 GPU-CPU gap。
- alpha=0.50 不是单调中点: CPU F1 反而最低，说明 S8 阈值/score 分布存在离散样本边界效应。
- 但此结果不能用于修改主表。它只能说明值得投入后续 AHL score 口径对齐，再决定是否正式重估 S8 alpha。

## 4. 给理论/规划线程的精华

1. 实验 A:
   - TF32 解释约 `6.9 ms` encoder 差异，速度比 `1.61x`。
   - 同步开销几乎不是主因，仅 `0.08-0.15 ms`。
   - 当前模型确认是 DINOv2 ViT-L/14 @224，batch=1 encoder-only 地板约 `11-18 ms`，未复现 6ms。

2. 实验 B:
   - P1/P2 均成立。
   - S8 alpha=0.70 在在线诊断口径下明显放大 GPU 退化。
   - alpha=0.35 方向有价值: CPU F1 0.8696，GPU F1 0.8613，dF1 -0.0083。
   - 结论必须保持 diagnostic only，不改 bridge v2 主表。

3. 对总体决策:
   - 时间线可以收束: encoder 地板不是 6ms，TF32/模型口径解释了大部分记忆偏差。
   - S8 退化有明确 bridge alpha 放大器嫌疑，但正式动作前要先做 AHL score 口径对齐。
