#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_ahl_plain_6_1_val_threshold_20260517/logs
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python export_qm_xiepai_clip_features.py \
  --dataset-root /gdata1/huangjd/xidun_mvtec_format_6_1 \
  --output-base /gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_clip_b16_val_threshold_cache \
  --plain-name plain_official_img_angle \
  --skip-chmm \
  --splits train val test \
  --device cuda:0 \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_ahl_plain_6_1_val_threshold_20260517/logs/export_plain_features.log
