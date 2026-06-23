# 20260609 CPU/GPU full-flow rerun v1 - PBS submission

Created at: 2026-06-09
Project: `/ghome/huangjd/code/detected/adpretrain_bridge`
Experiment: `20260609_cpu_gpu_full_flow_rerun_v1`

## Purpose

Rerun the full ADP-only, AHL-only, and ADP-AHL bridge flow under two preprocessing backends:

- CPU/PIL preprocessing.
- GPU `gpu_tensor_uint8_aa_true` preprocessing.

The decision target is whether GPU preprocessing can replace CPU preprocessing exactly or within acceptable error. Thresholds are selected on calib only; test is only used for final reporting.

## Jobs

| backend | job id | PBS file | run script | resource |
|---|---|---|---|---|
| CPU/PIL | `211302.Ghead` | `pbs/generated_submit/pbs_20260609_cpu_pil_full_flow_rerun_v1.pbs` | `pbs/generated_run/run_20260609_cpu_pil_full_flow_rerun_v1.sh` | `nodes=1:gpus=1:a` |
| GPU uint8 | `211303.Ghead` | `pbs/generated_submit/pbs_20260609_gpu_uint8_full_flow_rerun_v1.pbs` | `pbs/generated_run/run_20260609_gpu_uint8_full_flow_rerun_v1.sh` | `nodes=1:gpus=1:a` |

## Output paths

CPU/PIL:

- Feature cache: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/plain_dino_large_norm_cpu_pil_rerun_20260609_v1`
- Stage roots: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/stage_roots_cpu_pil_rerun_20260609_v1`
- ADP-only output: `output/20260609_cpu_gpu_full_flow_rerun_v1/cpu_pil_adp_only`
- AHL output: `output/20260609_cpu_gpu_full_flow_rerun_v1/cpu_pil_ahl`
- PBS logs: `pbs/20260609_cpu_pil_full_flow_rerun_v1.out`, `pbs/20260609_cpu_pil_full_flow_rerun_v1.err`

GPU uint8:

- Feature cache: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/plain_dino_large_norm_gpu_uint8_rerun_20260609_v1`
- Stage roots: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/stage_roots_gpu_uint8_rerun_20260609_v1`
- ADP-only output: `output/20260609_cpu_gpu_full_flow_rerun_v1/gpu_uint8_adp_only`
- AHL output: `output/20260609_cpu_gpu_full_flow_rerun_v1/gpu_uint8_ahl`
- PBS logs: `pbs/20260609_gpu_uint8_full_flow_rerun_v1.out`, `pbs/20260609_gpu_uint8_full_flow_rerun_v1.err`

## Fixed settings

- Split: `splits/20260529_qm_xiepai_6_1_fixed_180_70_val49`
- Backbone: `dinov2-large`
- ADP `num_ref=8`
- AHL `n_ref=5`
- AHL training: `epochs=30`, `steps_per_epoch=20`, `batch_size=48`, `seed=20260517`
- Bridge alpha: S1-S2=0.70, S3-S4=0.60, S5-S7=0.35, S8=0.70
- Threshold policy: stage recall best-F1 on calib; test report only.

## Follow-up aggregation

After both jobs finish, run on the login node:

```bash
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python summarize_cpu_gpu_full_flow.py \
  --cpu-adp-output output/20260609_cpu_gpu_full_flow_rerun_v1/cpu_pil_adp_only \
  --cpu-ahl-output output/20260609_cpu_gpu_full_flow_rerun_v1/cpu_pil_ahl \
  --gpu-adp-output output/20260609_cpu_gpu_full_flow_rerun_v1/gpu_uint8_adp_only \
  --gpu-ahl-output output/20260609_cpu_gpu_full_flow_rerun_v1/gpu_uint8_ahl \
  --out-dir summary/20260609_cpu_gpu_full_flow_rerun_v1
```
