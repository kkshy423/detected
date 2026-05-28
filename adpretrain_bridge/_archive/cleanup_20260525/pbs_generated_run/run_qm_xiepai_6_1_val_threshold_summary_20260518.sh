#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_6_1_val_threshold_summary_20260518/logs
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python summarize_val_threshold_results.py \
  --ahl-output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_ahl_plain_6_1_val_threshold_20260518 \
  --yolo-output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_6_1_val_threshold_20260517 \
  --yolo-retry-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_6_1_val_threshold_20260518_retry_s4_ampoff \
  --output-dir /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_6_1_val_threshold_summary_20260518 \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_6_1_val_threshold_summary_20260518/logs/summarize.log
