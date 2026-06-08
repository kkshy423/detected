# 激活环境
conda activate py310

# 定义全局路径变量 (请依你实际情况填写)
DATA_PATH="./benchmarks"   
CLASS_NAME="carpet"
MODEL_DIR="./trained_models"

# 1. 提取基础/干净图像的特征（用于源图像比对），包含 train 与 test
python datasets/plain_feature_extraction.py \
    --dataset_root $DATA_PATH \
    --classname $CLASS_NAME \
    --experiment_dir $MODEL_DIR \
    --model_name DRA

# 2. 提取 CutMix 增强数据对应的特征 (只操作 train)
python datasets/cutmix_feature_extraction.py \
    --dataset_root $DATA_PATH \
    --classname $CLASS_NAME \
    --experiment_dir $MODEL_DIR

# 3. 提取 CutPaste 增强数据对应的特征 (只操作 train)
python datasets/cutpaste_feature_extraction.py \
    --dataset_root $DATA_PATH \
    --classname $CLASS_NAME \
    --experiment_dir $MODEL_DIR

# 4. 提取 DREAM 增强数据对应的特征 (只操作 train)
python datasets/DREAM_feature_extraction.py \
    --dataset_root $DATA_PATH \
    --classname $CLASS_NAME \
    --model_dir $MODEL_DIR
