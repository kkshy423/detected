#!/bin/bash
set -euo pipefail

DIR_PATH=/ghome/huangjd/code/detected/AHL
DATA_PATH=/gdata1/huangjd/data/xidun_mvtec_format
MODEL_BASE=/ghome/huangjd/code/detected/DRA/experiment/xidun

cd "$DIR_PATH"
. /ghome/huangjd/.bashrc
conda activate ahl
export PYTHONUNBUFFERED=1

for class in models__球面斜拍 qiusai__models__球面俯拍  yuanzhu__models__孔口 qiusai__models__底面检测__端面检测  yuanzhu__models__内孔中   yuanzhu__models__端面
do
  MODEL_DIR="$MODEL_BASE/DRA_$class"

  echo "================ 开始特征提取类别: $class ================"

  # 1. 提取基础/干净图像特征（train + test）
  PYTHONPATH=. python -u datasets/plain_feature_extraction.py \
    --dataset_root "$DATA_PATH" \
    --classname "$class" \
    --experiment_dir "$MODEL_DIR" \
    --model_name DRA
  printf "基础特征提取完成: $class\n"
  # 2. 提取 CutMix 增强特征（仅 train）
  PYTHONPATH=. python -u datasets/cutmix_feature_extraction.py \
    --dataset_root "$DATA_PATH" \
    --classname "$class" \
    --experiment_dir "$MODEL_DIR"
  printf "CutMix 特征提取完成: $class\n"
  # 3. 提取 CutPaste 增强特征（仅 train）
  PYTHONPATH=. python -u datasets/cutpaste_feature_extraction.py \
    --dataset_root "$DATA_PATH" \
    --classname "$class" \
    --experiment_dir "$MODEL_DIR"
  printf "CutPaste 特征提取完成: $class\n"
  # 4. 提取 DREAM 增强特征（仅 train）
  PYTHONPATH=. python -u datasets/DREAM_feature_extraction.py \
    --dataset_root "$DATA_PATH" \
    --classname "$class" \
    --model_dir "$MODEL_DIR"
  printf "DREAM 特征提取完成: $class\n"
done
