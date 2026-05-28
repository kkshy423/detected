#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_threshold_compare_fixed_180_79_stage_v3/logs
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python evaluate_threshold_policy_variants.py \
  --source AHL_plain_ADPretrain=/ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_ahl_plain_fixed_180_79_stage_v3 \
  --source YOLO26s_cls=/ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_yolo26s_cls_fixed_180_79_stage_v3 \
  --output-dir /ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_threshold_compare_fixed_180_79_stage_v3 \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_threshold_compare_fixed_180_79_stage_v3/logs/evaluate_threshold_policy_variants.log
