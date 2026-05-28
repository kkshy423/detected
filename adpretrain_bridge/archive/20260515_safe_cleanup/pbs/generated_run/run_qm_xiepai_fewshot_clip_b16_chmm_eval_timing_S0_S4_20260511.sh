#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511/stages/S0/logs
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511/stages/S4/logs
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python evaluate_fewshot_stage_metrics.py \
  --stage S0 \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511 \
  --ahl-subdir ahl_retry4 \
  --benchmark-model-inference \
  --device cuda \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511/stages/S0/logs/evaluate_metrics_retry4_timingfix.log
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python evaluate_fewshot_stage_metrics.py \
  --stage S4 \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511 \
  --ahl-subdir ahl \
  --benchmark-model-inference \
  --device cuda \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511/stages/S4/logs/evaluate_metrics_retry4_timingfix.log
