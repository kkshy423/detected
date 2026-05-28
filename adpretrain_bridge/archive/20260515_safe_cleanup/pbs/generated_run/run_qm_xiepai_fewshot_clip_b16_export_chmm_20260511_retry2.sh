#!/usr/bin/env bash
set -euo pipefail
cd /ghome/huangjd/code/detected/adpretrain_bridge
mkdir -p /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511/logs
python3 export_qm_xiepai_clip_features.py   --output-base /gdata1/huangjd/data/xidun_qm_xiepai_adpretrain_clip_b16_fewshot_cache   --device cuda:0   2>&1 | tee /ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511/logs/feature_export_chmm_retry2.log
