#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511/stages/S0/logs
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python run_fewshot_ahl_stage.py \
  --stage S0 \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511 \
  --cache-root /gdata1/huangjd/data/xidun_qm_xiepai_adpretrain_clip_b16_fewshot_cache/chmm_official_img_angle_draref \
  --stage-root-base /gdata1/huangjd/data/xidun_qm_xiepai_adpretrain_clip_b16_fewshot_cache/stage_roots_chmm_official_img_angle_draref \
  --epochs 30 \
  --cluster-num 2 \
  --ahl-subdir ahl_retry4 \
  --reuse-stage-root \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511/stages/S0/logs/run_ahl_retry4.log
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python evaluate_fewshot_stage_metrics.py \
  --stage S0 \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511 \
  --ahl-subdir ahl_retry4 \
  --benchmark-model-inference \
  --device cuda \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511/stages/S0/logs/evaluate_metrics_retry4.log
