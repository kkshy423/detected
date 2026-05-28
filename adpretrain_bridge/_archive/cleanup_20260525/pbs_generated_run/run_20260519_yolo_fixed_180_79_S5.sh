#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_yolo26s_cls_fixed_180_79_stage_v3/stages/S5/logs
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python run_qm_xiepai_yolo26s_cls_val_threshold_stage.py \
  --stage S5 \
  --data-root /gdata1/huangjd/data/20260519_yolo_qm_xiepai_6_1_fixed_180_79_cls \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_yolo26s_cls_fixed_180_79_stage_v3 \
  --model /ghome/huangjd/code/detected/adpretrain_bridge/yolo26s-cls.pt \
  --epochs 200 \
  --imgsz 224 \
  --batch 16 \
  --device 0 \
  --workers 4 \
  --seed 20260519 \
  --patience 50 \
  --primary-policy strategy_stage_v3 \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_yolo26s_cls_fixed_180_79_stage_v3/stages/S5/logs/run_yolo.log
