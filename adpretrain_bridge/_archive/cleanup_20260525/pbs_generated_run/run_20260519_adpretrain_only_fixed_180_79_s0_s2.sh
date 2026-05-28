#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
OUT=/ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_adpretrain_only_fixed_180_79_s0_s2
mkdir -p "$OUT/logs"
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python evaluate_qm_xiepai_adpretrain_only_fixed_180_79.py \
  --split-root /ghome/huangjd/code/detected/adpretrain_bridge/splits/20260519_qm_xiepai_6_1_fixed_180_79 \
  --output-root "$OUT" \
  --source-parent /gdata1/huangjd/xidun_mvtec_format_6_1 \
  --adpretrain-root /ghome/huangjd/code/detected/ADPretrain \
  --checkpoint /ghome/huangjd/code/detected/ADPretrain/checkpoints/clip-base/checkpoints_img_angle.pth \
  --backbone clip-base \
  --num-ref 8 \
  --device cuda:0 \
  --stages S0 S1 S2 \
  2>&1 | tee "$OUT/logs/evaluate_adpretrain_only_fixed_180_79_s0_s2.log"
