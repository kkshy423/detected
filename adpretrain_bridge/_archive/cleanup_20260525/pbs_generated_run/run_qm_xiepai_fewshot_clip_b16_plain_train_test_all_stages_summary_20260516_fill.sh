#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_plain_train_test_20260512/metrics/fewshot_curve/logs
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python summarize_fewshot_curve_train_test.py \
  --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_plain_train_test_20260512 \
  --stages S0 S1 S2 S3 S4 S5 S6 S7 S8 \
  --main-doc /ghome/huangjd/code/detected/adpretrain_bridge/qm_xiepai_fewshot_clip_b16_plain_train_test_all_stages_20260516.md \
  2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_plain_train_test_20260512/metrics/fewshot_curve/logs/summarize_all_stages_20260516.log
