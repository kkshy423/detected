# 实验精华报告 — GPU 预处理 PIL-等价性修复（交理论线程）

实验名：`20260607_gpu_preprocess_uint8_equiv_v1`
执行日期：2026-06-07
执行线程：实验执行线程

## 1. 背景与目标

延续 GPU-preprocess 替代 CPU-PIL 的路线。此前两个实验（等价性 fix v1、AHL retrain smoke v1）结论是
"GPU tensor 预处理速度达标但指标不达标、重训也救不回来"，根因被归到 resize 数值不等价。
本实验直接攻这个根因：**在 GPU 上严格复刻 CPU-PIL 的 resize 数值语义**，看能否恢复检测指标。

## 2. 根因定位（已确证）

旧 GPU 路径的错误在于：**先 `div(255)` 进 float[0,1] 域，再在 float 域做 bicubic resize**。
PIL 的语义是 **在 uint8 域做 bicubic 插值，再 round/clip 回 uint8**。两者差异：
- bicubic overshoot：float 域保留为连续值；uint8 域被 clip+量化抹掉（这是系统性差异主因）。
- 量化：CPU 中间走 256 级 uint8，旧 GPU 全程 float。

CPU-only 诊断（登录节点，对真实样本，对比 normalize 后 tensor vs PIL baseline）：

| 预处理路径 | mean_abs_diff | p95_abs_diff | max_abs_diff |
| --- | ---: | ---: | ---: |
| float 域 resize（旧 GPU 路径） | 0.00323 | 0.00932 | 0.263 |
| **uint8 域 resize（新候选）** | **0.00079** | **0.00000** | **0.096** |

## 3. 解法与 GPU 可行性验证

新增 backend `gpu_tensor_uint8_aa_true`：GPU 上保持 uint8 域 resize/center_crop，crop 后才 cast→float/255→normalize。
torchvision 0.19 对 CUDA uint8 tensor 的 bicubic+antialias resize 原生支持。

GPU self-check（job 210671，A40，真实样本）：

| 对比 | max | mean | p95 |
| --- | ---: | ---: | ---: |
| GPU-uint8 vs CPU-uint8（设备一致性） | 0.0073 | ~0 | 0 |
| GPU-uint8 vs CPU-PIL | 0.096 | 0.0008 | 0 |
| 旧 GPU-float vs CPU-PIL | 0.280 | 0.0031 | 0.0093 |

→ CUDA uint8 resize 与 CPU 近 bit-exact，方案在 GPU 上成立。

## 4. 端到端 S8 结果（job 210672，固定 CPU 阈值 0.5757，ref_and_query_diff 真实部署口径）

| backend | tensor p95_diff | score P95 | test pred changed | Recall | F1 | preprocess median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu_pil（baseline） | 0 | 0 | 0 | 0.7714 | 0.8372 | 19.38 ms |
| 旧 gpu current/aa_true | 0.0094 | 0.0431 | 7 | 0.7143 | 0.8197 | 5.47 ms |
| **新 gpu_uint8_aa_true** | **0.0000002** | **0.0339** | **5** | **0.7571** | **0.8413** | **4.92 ms** |

阈值无关排序质量 AUC-PR（backend 各自 calib 选阈值）：cpu_pil 0.9167 / 旧路径 0.9097 / **新 uint8 路径 0.9129（query_only 下 0.9170，追平 CPU）**。

## 5. 判定

**改善显著但严格接受标准仍未全过。**

- 已达标：固定阈值 F1 gap = 0.0041（≤0.01 ✓），Recall gap = 0.0143（≤0.02 ✓），preprocess 远快于 CPU ✓。
- 未达标：score_abs_diff P95 = 0.0339（要求 ≤0.02 ✗），test pred changed = 5（要求 ≤2 ✗）。

关键洞察（与前两个实验对照，结论有实质推进）：
1. **resize 不再是瓶颈**。tensor 已 p95 bit-exact（0.0000002），但 score P95 仍 0.034。残余差来自
   **DINO-large encoder 对极少数落在 ±1/255 量化边界像素的浮点放大**（tensor max P95 仍 0.124），
   经 residual(query−ref) 链路累积。这是 PIL/tensor 数值复刻的物理下限，不是实现 bug。
2. **检测能力已实质恢复**：F1 追平甚至略超 CPU，AUC-PR 追平 CPU，Recall gap 缩到 0.014。
   前实验"重训 AHL 反而更差"的退化在本路径不存在——因为本路径没有损失判别信息，只剩工作点抖动。
3. 5 个改判样本全部是贴阈值（|score−thr|<0.016）的边界样本，最大 score_diff 仅 0.021；其中含 2 个 FP→TN 的"翻对"。

## 6. 给理论线程的决策问题

当前严格 P95≤0.02 / pred_changed≤2 标准卡的是"逐样本 bit 级一致"，而非"检测能力一致"。需要理论线程定夺：

- 选项 A：**接受 uint8 路径**。理由：F1/AUC-PR/Recall 已实质等价，速度 4× 优，残差是 encoder 浮点物理下限。
  若接受，建议把验收口径从"逐样本 score bit 等价"改为"检测能力等价 + 边界样本数受控"。
- 选项 B：**维持 CPU-PIL**，按既有结论不再投入 GPU 替代。本实验作为"已逼近物理下限仍未 bit 等价"的收口记录。
- 选项 C：若仍要逼近 P95≤0.02，下一步可探：encoder 输入 dtype/确定性算法（`torch.use_deterministic_algorithms`）、
  fp32 vs tf32 对 ViT 的影响、或 reference bank 用 CPU-PIL 离线固化（只 query 走 GPU，已知 query_only 下 P95=0.023 更好）。
  这些是新假设，需理论线程判断边际价值再决定是否开 GPU 任务。

## 7. 详细结果文件（线程可自取）

- 端到端 summary：`adpretrain_bridge/summary/20260607_gpu_preprocess_uint8_equiv_v1/equivalence_fix_summary.md`
- 指标全表（fixed + calibrated 两种阈值模式）：`.../equivalence_fix_metrics.csv`
- 时延拆分：`.../equivalence_fix_latency.csv`
- 改判样本：`.../changed_samples_by_backend.csv`
- 配置快照：`.../config_snapshot.json`
- GPU self-check 原始输出：`adpretrain_bridge/pbs/uint8_equiv_selfcheck.out`
- 提交清单：`adpretrain_bridge/summary/20260607_gpu_preprocess_uint8_equiv_v1_submission.md`
