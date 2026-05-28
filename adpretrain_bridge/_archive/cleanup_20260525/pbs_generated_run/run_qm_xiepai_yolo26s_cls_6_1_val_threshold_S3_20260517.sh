#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_6_1_val_threshold_20260517/stages/S3/logs
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python run_qm_xiepai_yolo26s_cls_val_threshold_stage.py \
  --stage S3 \
  --data-root /gdata1/huangjd/data/yolo_qm_xiepai_6_1_cls_val_threshold_20260517 \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_6_1_val_threshold_20260517 \
  --model /ghome/huangjd/code/detected/adpretrain_bridge/yolo26s-cls.pt \
  --epochs 200 \
  --imgsz 224 \
  --batch 16 \
  --device 0 \
  --workers 4 \
  --seed 20260517 \
  --patience 50 \
  --primary-policy val_best_f1 \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_6_1_val_threshold_20260517/stages/S3/logs/run_yolo.log
