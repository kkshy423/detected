#!/bin/bash
set -euo pipefail

DIR_PATH=/ghome/huangjd/code/detected/AHL
DATA_PATH=/gdata1/huangjd/data/xidun_mvtec_format
AHL_SAVE_BASE=/ghome/huangjd/code/detected/AHL/trained_models_xidun

cd "$DIR_PATH"
. /ghome/huangjd/.bashrc
conda activate ahl
export PYTHONUNBUFFERED=1

#for class in bottle grid screw toothbrush cable capsule carpet hazelnut leather metal_nut pill tile transistor wood zipper
for class in yuanzhu__models__端面
do
  AHL_SAVE_DIR="$AHL_SAVE_BASE/AHL_MVTec_$class"
  mkdir -p "$AHL_SAVE_DIR"

  echo "================ 开始 AHL 训练类别: $class ================"

  PYTHONPATH=. python -u main.py \
    --dataset_root "$DATA_PATH" \
    --classname "$class" \
    --feat_classname "$class" \
    --experiment_dir "$AHL_SAVE_DIR" \
    --model_name DRA

  printf "AHL 训练完成: %s\n" "$class"
done
