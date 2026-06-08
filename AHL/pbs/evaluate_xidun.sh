#!/bin/bash
set -euo pipefail

DIR_PATH=/ghome/huangjd/code/detected/AHL
DATA_PATH=/gdata1/huangjd/data/xidun_mvtec_format
AHL_SAVE_BASE=/ghome/huangjd/code/detected/AHL/trained_models_xidun

cd "$DIR_PATH"
. /ghome/huangjd/.bashrc
conda activate ahl
python /ghome/huangjd/code/detected/AHL/pbs/evaluate_xidun_aligned_metrics.py --benchmark-model-inference