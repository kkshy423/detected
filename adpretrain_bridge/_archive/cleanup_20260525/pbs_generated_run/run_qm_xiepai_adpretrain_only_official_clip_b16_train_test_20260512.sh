#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_adpretrain_only_clip_b16_official_train_test_20260512/logs
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python evaluate_qm_xiepai_adpretrain_only_official_train_test.py \
  --split-root /ghome/huangjd/code/detected/adpretrain_bridge/splits/qm_xiepai_fewshot_train_test \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_adpretrain_only_clip_b16_official_train_test_20260512 \
  --source-class-root /gdata1/huangjd/data/xidun_mvtec_format/models__球面斜拍 \
  --adpretrain-root /ghome/huangjd/code/detected/ADPretrain \
  --checkpoint /ghome/huangjd/code/detected/ADPretrain/checkpoints/clip-base/checkpoints_img_angle.pth \
  --backbone clip-base \
  --num-ref 8 \
  --device cuda:0 \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_adpretrain_only_clip_b16_official_train_test_20260512/logs/evaluate_official_adpretrain_only.log