# 20260608 encoder latency breakdown and S8 alpha probe v1 - PBS submission

Created at: 2026-06-09
Project: `/ghome/huangjd/code/detected/adpretrain_bridge`
Experiment: `20260608_encoder_latency_breakdown_and_s8_alpha_probe_v1`

## Purpose

Two independent diagnostic experiments:

- Experiment A: diagnose why encoder latency is tens of ms rather than the remembered 6 ms.
- Experiment B: probe whether S8 bridge alpha=0.70 amplifies CPU/GPU gap. Diagnostic only, not a main-table update.

## Jobs

| experiment | job id | PBS file | run script | resource |
|---|---|---|---|---|
| A encoder latency breakdown | `211466.Ghead` | `pbs/generated_submit/pbs_20260608_encoder_latency_breakdown_v1.pbs` | `pbs/generated_run/run_20260608_encoder_latency_breakdown_v1.sh` | `nodes=1:gpus=1:a` |
| B S8 alpha probe diagnostic | `211467.Ghead` | `pbs/generated_submit/pbs_20260608_s8_alpha_probe_v1.pbs` | `pbs/generated_run/run_20260608_s8_alpha_probe_v1.sh` | `nodes=1:gpus=1:a` |

## Fixed settings

- GPU: A40 queue/resource line unchanged, `#PBS -l nodes=1:gpus=1:a`
- Python: `/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python`
- Seed: `0`
- Batch: `1`
- A input backend: `cpu_pil`, predecoded/preprepared tensors before encoder timing
- B backends: `cpu_pil`, `gpu_tensor_uint8_aa_true`
- B stage: `S8` only
- B alphas: `0.70`, `0.50`, `0.35`
- B threshold: selected on calib, test reported only

## Expected outputs

Experiment A:

- `summary/20260608_encoder_latency_breakdown_and_s8_alpha_probe_v1/encoder_latency_breakdown.csv`
- `summary/20260608_encoder_latency_breakdown_and_s8_alpha_probe_v1/encoder_latency_breakdown.md`
- `summary/20260608_encoder_latency_breakdown_and_s8_alpha_probe_v1/encoder_latency_backbone_meta.json`

Experiment B:

- `summary/20260608_encoder_latency_breakdown_and_s8_alpha_probe_v1/s8_alpha_probe_diagnostic_not_main_table_update.csv`
- `summary/20260608_encoder_latency_breakdown_and_s8_alpha_probe_v1/s8_alpha_probe_gap_diagnostic_not_main_table_update.csv`
- `summary/20260608_encoder_latency_breakdown_and_s8_alpha_probe_v1/s8_alpha_probe_diagnostic_not_main_table_update.md`

## Hard boundary for B

This is diagnostic only. Do not update bridge v2 main table from this result. Main-table alpha changes require AHL score口径对齐 first, which is out of scope here.
