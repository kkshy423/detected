# Copyright (c) 2026 Sapphire9877@gmail.com
# This document records the analysis of the AHL project's README steps and their mapping to the research paper.

# 任务日志 (Behavior Log)

## 2026-04-13: README 与 论文方法映射分析

### 任务目标
解析 `README.md` 中的运行步骤，并将其与论文《Anomaly Heterogeneity Learning for Open-set Supervised Anomaly Detection》中的理论方法进行对应。

### 执行过程
1.  **检索与阅读**：阅读了项目根目录下的 `README.md`。
2.  **理论分析**：阅读了用户提供的论文 Markdown 文本，重点关注了 3.2 节（HADG）和 3.3 节（CDL）。
3.  **代码核对**：
    *   检查了 `main.py` 中的训练逻辑。
    *   检查了 `datasets/` 目录下的特征提取脚本。
    *   核实了基准模型（DRA/DevNet）的调用方式。
    *   核实了基准模型（DRA/DevNet）的调用方式。
4.  **建立映射**：总结了 4 个主要步骤的具体功能及其在论文算法流程中的位置。
5.  **代码重构与兼容性修复**：
    *   创建并完善了 `modeling/net_DRA.py` 以支持特征提取。
    *   通过引入 `extracted` 和 `flag` 参数解决了不同提取脚本之间的接口不一致问题（分别适配 CutMix/CutPaste 与 DREAM 脚本）。
    *   创建了 `datasets/plain_feature_extraction.py` 以补全原始特征提取功能。
    *   适配了用户环境路径：`--dataset_root ./benchmarks` 和 `./trained_models/model_dra_carpet.pkl`。


---

# 项目使用文档 (Project Documentation)

## AHL (Anomaly Heterogeneity Learning) 运行流程解析

本方案基于对 AHL 官方实现代码的分析，阐述了运行该项目的四个关键阶段及其背后的理论支持。

### 1. 数据准备 (Dataset Setup)
*   **功能**：标准化数据集结构。
*   **操作**：下载 MVTec AD 或其他支持的异常检测数据集，并使用脚本转换为特定的目录格式（`train/good`, `test/defect_class_x`）。
*   **论文对应**：Section 4.1 (Experimental Setup - Datasets)。
*   **重要性**：HADG 组件依赖于对正常类数据的精确聚类（$C$ clusters），标准化的格式确保了聚类和伪异常注入的正确性。

### 2. 基准模型训练 (Base Model Pre-training)
*   **功能**：获得基础特征提取能力。
*   **操作**：运行原有的 OSAD 模型（如 DRA 或 DevNet）并保存其权重文件（`.pkl`）。
*   **论文对应**：Section 3.1 & 4.1.
*   **说明**：AHL 是一个“即插即用”的框架。它不从零开始，而是利用 DRA/DevNet 提取的低维特征作为输入。保存的权重将作为 AHL 统一模型 $g$ 的初始化状态。

### 3. 特征缓存与分布生成 (Feature & Augmentation Saving)
*   **功能**：模拟异构异常分布并加速训练。
*   **操作**：运行 `datasets/` 下的特征提取脚本（如 `DREAM_feature_extraction.py`），生成并保存正常样本及增强后的伪异常特征。
*   **论文对应**：Section 3.2 **HADG (Heterogeneous Anomaly Distribution Generation)**。
*   **方法论**：HADG 通过 CutMix、CutPaste 和 DREAM 产生异构的负样本。通过预先计算好这些特征，可以避免在主循环中重复进行耗时的图像增强和推理。

### 4. 协同微分学习训练 (Running AHL Core)
*   **功能**：执行 AHL 核心算法。
*   **命令示例**：
    ```bash
    python main.py --dataset_root $DATA_PATH --classname $CLASS --feat_classname $FEAT_CLASS --experiment_dir $SAVE_PATH
    ```
*   **论文对应**：Section 3.3 **CDL (Collaborative Differentiable Learning)** 与 **Sequential Modeling ($\psi$)**。
*   **核心细节**：
    *   **CDL 训练**：`main.py` 会创建多个 Episode（Episode 数量对应 $T$）。在每个 Episode 内进行 Support 集微调和 Query 集验证。
    *   **动态权重**：代码中的 `AUX_Model`（Bi-LSTM）会根据历史 Epoch 的得分变化，动态调整不同异构分布对最终模型更新的贡献度（重要性得分 $w_i^t$）。

---
