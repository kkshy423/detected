# AHL / YOLO 可对齐指标说明

## 指标要求

- 分类模型准确率 ≥ 98%
- 目标检测模型 mAP ≥ 95%
- 语义分割与实例分割模型 IoU ≥ 0.7
- 单张图像推理时间 ≤ 50ms

## AHL

- 可以对齐：分类准确率、Precision、Recall、F1、AUC-ROC、AUC-PR
- 部分可以对齐：单张图像推理时间
- 不可以对齐：目标检测 mAP、分割 IoU

### 原因

- AHL 当前产物是图像级异常分数 `result.txt`，适合做二分类阈值化评估。
- AHL 不输出检测框，也不输出像素级 mask，因此无法直接计算检测 mAP 与分割 IoU。
- 若使用 `--benchmark-model-inference`，可以测量特征级 AHL 模型 forward 时间；但它不含 DRA 特征提取与磁盘 I/O，属于端到端耗时的下界参考。

## YOLO11

- 可以对齐：分类准确率、Precision、Recall、F1、AUC-ROC、AUC-PR、检测 mAP、分割 IoU、单张图像推理时间

### 原因

- YOLO 已有 `results.csv`，可直接取最后一轮的 `mAP50(B)` / `mAP50-95(B)`。
- YOLO 运行时会输出实例 mask，可与 GT polygon rasterize 后计算 defect-image union IoU。
- YOLO 可用图像级异常分数（最大框置信度）做 good / known 二分类，从而得到 Accuracy / Precision / Recall / F1 / AUC。
- YOLO 的 `result.speed` 可直接统计单张图像的 preprocess / inference / postprocess 耗时。

## 两套系统仍然不完全同协议的地方

- AHL 的分类指标来自图像级异常分数阈值化；YOLO 的分类指标也是脚本里派生出的图像级异常分数阈值化。
- 这两者可以横向对比，但它们都不是 YOLO 原生检测验证中的 `precision_M / recall_M`。
- 若看严格同协议横比，建议优先比较两边脚本生成的图像级 Accuracy / Precision / Recall / F1 / AUC。
