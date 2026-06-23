# 20260609 CPU/GPU 完整流程重跑 v1 - 阶段交接

时间: 2026-06-09
项目: `/ghome/huangjd/code/detected/adpretrain_bridge`
实验 ID: `20260609_cpu_gpu_full_flow_rerun_v1`

## 1. 方法关系与原理

本轮要比较的是“预处理是否可替换”，不是重新调参。

- ADP-only-DINO: training-free 残差匹配基线。用正常参考图建立 DINO/projector 残差参考库，对 query 做 patch 级最近邻匹配，残差图经过固定公式得到 image score。本项目 ADP 固定 `num_ref=8`。
- AHL-DINO: 在 ADP/DINO 投影残差 feature cache 上训练 few-shot 检测器。AHL 的 reference 语义不同，固定 `n_ref=5`，训练参数 `epochs=30`, `steps_per_epoch=20`, `batch_size=48`, `seed=20260517`。
- ADP-AHL-bridge-v2: 后融合，不再训练模型。每个 stage 上用 calib normal score 分别对 ADP/AHL 做 robust normalization，再按固定 alpha 融合: S1-S2=0.70, S3-S4=0.60, S5-S7=0.35, S8=0.70；S0 无 bridge。
- 阈值策略: 只用 calib 选阈值，test 只报告。S0-S2 用 best-F1@R>=0.90，S3-S6 用 best-F1@R>=0.85，S7-S8 用纯 best-F1。禁止 test oracle。

因此 CPU 与 GPU 的受控差异只应是预处理:

- CPU: CPU/PIL 预处理。
- GPU: `gpu_tensor_uint8_aa_true`，即 GPU 上保留 uint8 域 resize/crop，crop 后再 cast/normalize。

## 2. 已完成动作

已新增并上传以下远程文件:

- `pbs/generated_run/run_20260609_cpu_pil_full_flow_rerun_v1.sh`
- `pbs/generated_run/run_20260609_gpu_uint8_full_flow_rerun_v1.sh`
- `pbs/generated_submit/pbs_20260609_cpu_pil_full_flow_rerun_v1.pbs`
- `pbs/generated_submit/pbs_20260609_gpu_uint8_full_flow_rerun_v1.pbs`
- `summarize_cpu_gpu_full_flow.py`
- `summary/20260609_cpu_gpu_full_flow_rerun_v1/relationship_and_plan.md`
- `summary/20260609_cpu_gpu_full_flow_rerun_v1/pbs_submission.md`

已提交 PBS:

| backend | job id | 状态 |
|---|---|---|
| CPU/PIL 完整流程 | `211302.Ghead` | 排队，`comment = Not Running: Not enough of the right type of nodes are available` |
| GPU uint8 完整流程 | `211303.Ghead` | 排队，`comment = Not Running: Not enough of the right type of nodes are available` |

资源行均为授权口径:

```text
#PBS -l nodes=1:gpus=1:a
```

## 3. 本轮重跑预期产物

CPU/PIL:

- Feature cache: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/plain_dino_large_norm_cpu_pil_rerun_20260609_v1`
- Stage roots: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/stage_roots_cpu_pil_rerun_20260609_v1`
- ADP-only: `output/20260609_cpu_gpu_full_flow_rerun_v1/cpu_pil_adp_only`
- AHL: `output/20260609_cpu_gpu_full_flow_rerun_v1/cpu_pil_ahl`

GPU uint8:

- Feature cache: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/plain_dino_large_norm_gpu_uint8_rerun_20260609_v1`
- Stage roots: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/stage_roots_gpu_uint8_rerun_20260609_v1`
- ADP-only: `output/20260609_cpu_gpu_full_flow_rerun_v1/gpu_uint8_adp_only`
- AHL: `output/20260609_cpu_gpu_full_flow_rerun_v1/gpu_uint8_ahl`

## 4. 当前已有产物 preliminary 结论

等待 PBS 期间，已用已有 CPU 20260529/v2 产物和 GPU 20260608 产物跑同一套汇总脚本，输出在:

- `summary/20260609_cpu_gpu_full_flow_rerun_v1/preliminary_existing_outputs/cpu_gpu_full_flow_summary.md`
- `summary/20260609_cpu_gpu_full_flow_rerun_v1/preliminary_existing_outputs/cpu_gpu_full_flow_master_table.csv`
- `summary/20260609_cpu_gpu_full_flow_rerun_v1/preliminary_existing_outputs/cpu_gpu_full_flow_gap_summary.csv`
- `summary/20260609_cpu_gpu_full_flow_rerun_v1/preliminary_existing_outputs/cpu_gpu_full_flow_fixed_cpu_threshold_table.csv`

这不是本轮重跑最终结果，只是目前已有证据。

preliminary 主要观察:

- ADP-only: GPU 与 CPU 基本等价。最大 `|dF1|=0.0392`，最大 `|dAUPR|=0.0104`；这是 training-free 口径下最可信的预处理替换信号。
- AHL-only: 波动明显，最大 `|dF1|=0.1587`，S0/S5/S8 等 stage 差异较大；不能直接归因给 GPU 预处理，必须看本轮 CPU 同流程重训相对 v2 的漂移。
- Bridge: 主生产口径多数 stage 差异较小，最大 `|dF1|=0.0686` 来自 S8；S1-S7 大多在小幅波动区间，S8 仍是重点风险 stage。

preliminary 表里的关键行:

| method | stage | CPU F1 | GPU F1 | dF1 | CPU R | GPU R | dR | CPU AUPR | GPU AUPR | dAUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ADP-only-DINO | S0-S2 | 0.7753 | 0.7582 | -0.0170 | 0.9857 | 0.9857 | 0.0000 | 0.9151 | 0.9047 | -0.0104 |
| ADP-only-DINO | S3-S6 | 0.7931 | 0.8323 | +0.0392 | 0.9857 | 0.9571 | -0.0286 | 0.9151 | 0.9047 | -0.0104 |
| ADP-only-DINO | S7-S8 | 0.8429 | 0.8406 | -0.0023 | 0.8429 | 0.8286 | -0.0143 | 0.9151 | 0.9047 | -0.0104 |
| ADP-AHL-bridge-v2 | S1 | 0.8092 | 0.8235 | +0.0143 | 1.0000 | 1.0000 | 0.0000 | 0.9324 | 0.9249 | -0.0075 |
| ADP-AHL-bridge-v2 | S4 | 0.9139 | 0.8874 | -0.0265 | 0.9857 | 0.9571 | -0.0286 | 0.9437 | 0.9333 | -0.0104 |
| ADP-AHL-bridge-v2 | S6 | 0.8873 | 0.9028 | +0.0155 | 0.9000 | 0.9286 | +0.0286 | 0.9471 | 0.9429 | -0.0042 |
| ADP-AHL-bridge-v2 | S8 | 0.8593 | 0.7907 | -0.0686 | 0.8286 | 0.7286 | -0.1000 | 0.9276 | 0.9360 | +0.0084 |

## 5. 等 PBS 完成后的下一步

两条 PBS 完成后，在登录节点运行纯 CPU 汇总:

```bash
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python summarize_cpu_gpu_full_flow.py \
  --cpu-adp-output output/20260609_cpu_gpu_full_flow_rerun_v1/cpu_pil_adp_only \
  --cpu-ahl-output output/20260609_cpu_gpu_full_flow_rerun_v1/cpu_pil_ahl \
  --gpu-adp-output output/20260609_cpu_gpu_full_flow_rerun_v1/gpu_uint8_adp_only \
  --gpu-ahl-output output/20260609_cpu_gpu_full_flow_rerun_v1/gpu_uint8_ahl \
  --out-dir summary/20260609_cpu_gpu_full_flow_rerun_v1
```

最终需要给理论/规划线程的判断:

1. ADP-only CPU/GPU 是否仍保持近等价。
2. AHL CPU 重跑 vs v2 的漂移是否与 GPU 重跑 vs CPU 重跑同量级。
3. bridge CPU/GPU 的 production-facing gap 是否可接受，尤其复查 S8。
4. 若 AHL CPU 重跑也明显偏离 v2，则 AHL 单方法差异应归为 few-shot 重训稳定性问题，GPU 不单独背锅。
