#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_train_test_20260512/stages/S8/logs
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python run_fewshot_ahl_stage_train_test.py \
  --stage S8 \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_train_test_20260512 \
  --split-root /ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_fewshot_train_test \
  --cache-root /gdata1/huangjd/data/xidun_qm_xiepai_adpretrain_clip_b16_fewshot_cache/chmm_official_img_angle_draref \
  --stage-root-base /gdata1/huangjd/data/xidun_qm_xiepai_adpretrain_clip_b16_fewshot_cache/stage_roots_chmm_official_img_angle_draref_train_test \
  --epochs 30 \
  --cluster-num 2 \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_train_test_20260512/stages/S8/logs/run_ahl.log
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python evaluate_fewshot_stage_metrics_train_test.py \
  --stage S8 \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_train_test_20260512 \
  --benchmark-model-inference \
  --device cuda \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_train_test_20260512/stages/S8/logs/evaluate_metrics.log
