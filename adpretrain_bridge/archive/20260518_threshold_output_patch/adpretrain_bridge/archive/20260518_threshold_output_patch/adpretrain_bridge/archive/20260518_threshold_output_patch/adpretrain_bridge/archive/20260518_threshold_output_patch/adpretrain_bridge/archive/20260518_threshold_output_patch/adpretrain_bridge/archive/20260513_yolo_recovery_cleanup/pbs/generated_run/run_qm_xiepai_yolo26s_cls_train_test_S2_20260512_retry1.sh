#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
export WANDB_DISABLED=true
export YOLO_CONFIG_DIR=/ghome/huangjd/code/detected/adpretrain_bridge/.ultralytics
export PIP_CACHE_DIR=/gdata1/huangjd/data/pip_cache
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260512_retry1/stages/S2/logs
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python run_qm_xiepai_yolo26s_cls_stage.py \
  --stage S2 \
  --data-root /gdata1/huangjd/data/yolo_qm_xiepai_cls_fewshot_train_test_20260512 \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260512_retry1 \
  --model yolo26s-cls.pt \
  --epochs 200 \
  --imgsz 224 \
  --batch 16 \
  --device 0 \
  --workers 4 \
  --seed 20260512 \
  --patience 50 \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260512_retry1/stages/S2/logs/run_yolo26s_cls.log
