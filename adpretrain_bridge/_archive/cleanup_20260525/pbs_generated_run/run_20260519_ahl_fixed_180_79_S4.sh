#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_ahl_plain_fixed_180_79_stage_v3/stages/S4/logs
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python run_fewshot_ahl_stage_val_threshold.py \
  --stage S4 \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_ahl_plain_fixed_180_79_stage_v3 \
  --split-root /ghome/huangjd/code/detected/adpretrain_bridge/splits/20260519_qm_xiepai_6_1_fixed_180_79 \
  --cache-root /gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_clip_b16_val_threshold_cache/plain_official_img_angle \
  --stage-root-base /gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_clip_b16_val_threshold_cache/20260519_stage_roots_plain_fixed_180_79 \
  --epochs 30 \
  --cluster-num 2 \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_ahl_plain_fixed_180_79_stage_v3/stages/S4/logs/run_ahl.log
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python evaluate_fewshot_stage_metrics_val_threshold.py \
  --stage S4 \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_ahl_plain_fixed_180_79_stage_v3 \
  --primary-policy strategy_stage_v3 \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_ahl_plain_fixed_180_79_stage_v3/stages/S4/logs/evaluate_metrics.log
