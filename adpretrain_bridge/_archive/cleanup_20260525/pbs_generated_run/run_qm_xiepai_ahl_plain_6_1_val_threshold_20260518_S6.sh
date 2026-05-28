#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_ahl_plain_6_1_val_threshold_20260518/stages/S6/logs
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python run_fewshot_ahl_stage_val_threshold.py \
  --stage S6 \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_ahl_plain_6_1_val_threshold_20260518 \
  --split-root /ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_6_1_val_threshold \
  --cache-root /gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_clip_b16_val_threshold_cache/plain_official_img_angle \
  --stage-root-base /gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_clip_b16_val_threshold_cache/stage_roots_plain_official_img_angle_val_threshold_20260518 \
  --epochs 30 \
  --cluster-num 2 \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_ahl_plain_6_1_val_threshold_20260518/stages/S6/logs/run_ahl.log
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python evaluate_fewshot_stage_metrics_val_threshold.py \
  --stage S6 \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_ahl_plain_6_1_val_threshold_20260518 \
  --primary-policy strategy_stage_adaptive \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_ahl_plain_6_1_val_threshold_20260518/stages/S6/logs/evaluate_metrics.log
