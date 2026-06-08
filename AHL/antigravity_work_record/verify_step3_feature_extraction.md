# Copyright (c) 2026 Sapphire9877@gmail.com

## 工作记录：验证并确认step3特征提取的正确性

**任务概述：**
用户要求根据`README.md`和论文的内容，并以`cutmix_feature_extraction.py`为准，研究并确保`plain_feature_extraction.py`、`cutpaste_feature_extraction.py`、`DREAM_feature_extraction.py`在`step3`中的特征提取逻辑对于后续的`step4`（AHL模型训练）能够正常工作。尽可能少地修改代码。

**分析与排查过程：**
1. **对比特征维度和文件格式：**
   * 以`cutmix_feature_extraction.py`作为参考标准，模型调用时使用`model(image = augmented_image.unsqueeze(0).cuda(), flag = True)`，因为`flag=True`时的返回值为一个长度为`n_scales`（通常为2）的特征列表：`[feature_s0, feature_s1]`。
   * 参考代码中通过对输出特征执行列表推导解析外层维度：`[item.cpu().detach().numpy() for item in feature]`，并使用`.squeeze()`，在单张图像（batch_size=1）处理的情况下获得了尺寸为`(C, H, W)`的NumPy张量，最后保存为`.npy`文件。
   * `cutpaste_feature_extraction.py`与`DREAM_feature_extraction.py`中的处理逻辑与`cutmix_feature_extraction.py`**完全一致**。
   * `plain_feature_extraction.py`由于读取了相同的统一模型列表输出，直接通过`features[0].cpu().detach().numpy().squeeze()`提取，这种处理由于都是获取特征图并移除batch维度，生成的数据格式和结果尺寸也**等效且一致**于`(C, H, W)`，可以直接供给AHL训练脚本`main.py`的`Task_Dataset`加载。

2. **验证文件存放的层次结构：**
   根据`README.md`，特征应保存在诸如：
   `DATA_PATH/subset_1/feature/train/`，`DATA_PATH/subset_1/aug_mix/train/`等路径中。
   * 修复了`plain_feature_extraction.py`的一个目录路径拼接错误：代码对`train/good`数据集提取时错误地拼接成了`feature/train/good/good`，导致路径层级不符合AHL期望。这一代码现已修正，`train/good`提取时生成路径为`feature/train/good`，`test/*`被正确保存到`feature/test/cls`。

**说明文档：修改后的`datasets/plain_feature_extraction.py`**
* **功能：** 用于在基础模型（DRA/DevNet）上抽取正常图像（`train/good`）及测试图像（`test/`）的多尺度特征。它的输出供给AHL算法框架用于特征空间中的判别性损失比对。
* **使用方法：** 
  ```bash
  python datasets/plain_feature_extraction.py --dataset_root /path/to/data --classname [your_class_name] --experiment_dir /path/to/base_model_weights
  ```
* **工作流影响：** 所有从`step3`生成的`.npy`文件由于尺寸对其且目录规范正确，现在能被`main.py`正确读取，作为表征传入到HolisticHead和CompositeHead等分类头中。

**附加内容**
本次未对`cutmix`、`cutpaste`和`DREAM`特征抽取作破坏性修改以防止失效，仅优化了`plain`抽取时的逻辑错误，所有脚本返回的`flag=True`特征输出保持完全相同的数据序列。

