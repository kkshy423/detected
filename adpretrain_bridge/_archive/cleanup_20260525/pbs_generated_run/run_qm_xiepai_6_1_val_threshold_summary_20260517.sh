#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_6_1_val_threshold_summary_20260517
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python summarize_val_threshold_results.py \
  --ahl-output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_ahl_plain_6_1_val_threshold_20260517 \
  --yolo-output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_6_1_val_threshold_20260517 \
  --output-dir /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_6_1_val_threshold_summary_20260517 \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_6_1_val_threshold_summary_20260517/summary.log
