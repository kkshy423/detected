#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge

export YOLO_PYTHON_DEPS_DIR=/gdata1/huangjd/data/yolo26s_cls_py38_headless_deps_20260513_retry3_nopolars
PYTHON=/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python
OUTPUT_ROOT=/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3_recovered_eval
mkdir -p "${OUTPUT_ROOT}/logs"

exec "${PYTHON}" recover_qm_xiepai_yolo26s_cls_retry3_eval.py \
  --data-root /gdata1/huangjd/data/yolo_qm_xiepai_cls_fewshot_train_test_20260512 \
  --source-output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3 \
  --output-root "${OUTPUT_ROOT}" \
  --stages S0 S1 S2 S3 S4 S5 S6 S7 S8 \
  --imgsz 224 \
  --device 0 \
  2>&1 | tee "${OUTPUT_ROOT}/logs/recover_yolo_retry3_eval.log"
