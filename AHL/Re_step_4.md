# 激活环境 (确保在正确环境中)
conda activate py310

# 定义路径变量
DATA_PATH="./benchmarks"   
CLASS_NAME="carpet"
AHL_SAVE_DIR="./trained_models_AHL"  # 为区别于Step2的基础模型，这里建议设置一个新的存储路径

# 执行 Step 4：基于提取好的特征，运行 AHL
python main.py \
    --dataset_root $DATA_PATH \
    --classname $CLASS_NAME \
    --feat_classname $CLASS_NAME \
    --experiment_dir $AHL_SAVE_DIR \
    --model_name DRA
