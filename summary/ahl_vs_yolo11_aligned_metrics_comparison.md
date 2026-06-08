# AHL 与 YOLO11 对齐指标对比

## 评估协议

- 本报告在可以对齐的前提下，按照统一的图像级异常检测协议比较 AHL 与 YOLO11。
- AHL 指标来自最终保存的 `result.txt` 分数；YOLO11 指标则使用每张图像预测框中的最大置信度作为图像级异常分数。
- YOLO11 的检测 mAP 和分割 IoU 作为 YOLO 专属结构化指标单独报告，因为 AHL 不输出检测框和分割 mask。

## 指标解释

| 指标 | 含义 | 是否越高越好 | 适用方法 |
| --- | --- | --- | --- |
| Accuracy | 对图像级异常分数设置阈值后得到的二分类准确率；正常样本记为负类，异常样本记为正类。 | 是 | AHL, YOLO11 |
| Precision | 在所有被预测为异常的图像中，真正异常图像所占的比例。Precision 越高，说明误报越少。 | 是 | AHL, YOLO11 |
| Recall | 在所有真实异常图像中，被成功检出的比例。Recall 越高，说明漏检越少。 | 是 | AHL, YOLO11 |
| F1 | Precision 与 Recall 的调和平均，用于综合平衡误报与漏报。 | 是 | AHL, YOLO11 |
| AUC-ROC | 基于真正例率与假正例率关系的无阈值排序指标。 | 是 | AHL, YOLO11 |
| AUC-PR | 基于 Precision-Recall 曲线的无阈值排序指标，通常对类别不平衡更敏感。 | 是 | AHL, YOLO11 |
| Inference ms | 单张图像平均推理耗时，单位为毫秒。这个指标越低越好。 | 否 | AHL, YOLO11 |
| mAP50(B) | YOLO 在 IoU=0.50 条件下的检测框 mAP。AHL 无法提供该指标，因为它不输出检测框。 | 是 | 仅 YOLO11 |
| IoU | 在异常图像上，预测 mask 与 GT mask 的 union-mask IoU 平均值。AHL 无法提供该指标，因为它不输出 mask。 | 是 | 仅 YOLO11 |

## 均值对比

| 方法 | 类别数 | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms | mAP50(B) | IoU |
| AHL | 6 | 0.8880 | 0.9040 | 0.9193 | 0.9069 | 0.9340 | 0.9653 | 12.6147 | N/A | N/A |
| YOLO11 | 6 | 0.6868 | 0.5940 | 0.8217 | 0.6475 | 0.6603 | 0.6185 | 33.2603 | 0.7410 | 0.8043 |

## 均值差异

| 指标 | AHL - YOLO11 | 更优方法 |
| --- | ---: | --- |
| Accuracy | 0.2012 | AHL |
| Precision | 0.3100 | AHL |
| Recall | 0.0976 | AHL |
| F1 | 0.2594 | AHL |
| AUC-ROC | 0.2737 | AHL |
| AUC-PR | 0.3468 | AHL |
| Inference ms | -20.6456 | AHL |

## 各子类图像级指标对比

| 子类 | AHL Acc | YOLO Acc | AHL Prec | YOLO Prec | AHL Recall | YOLO Recall | AHL F1 | YOLO F1 | AHL AUC-ROC | YOLO AUC-ROC | AHL AUC-PR | YOLO AUC-PR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| models__球面斜拍 | 0.8663 | 0.9389 | 0.8796 | 0.8919 | 0.8889 | 0.8250 | 0.8842 | 0.8571 | 0.9420 | 0.9361 | 0.9621 | 0.8977 |
| qiusai__models__底面检测__端面检测 | 0.7546 | 0.8087 | 0.7471 | 0.7879 | 1.0000 | 0.6341 | 0.8553 | 0.7027 | 0.8512 | 0.7370 | 0.9482 | 0.7183 |
| qiusai__models__球面俯拍 | 0.9337 | 0.6667 | 0.9607 | 0.6667 | 0.9661 | 1.0000 | 0.9634 | 0.8000 | 0.9227 | 0.5014 | 0.9910 | 0.7514 |
| yuanzhu__models__内孔中 | 0.9726 | 0.6970 | 1.0000 | 0.5789 | 0.9623 | 0.8462 | 0.9808 | 0.6875 | 0.9943 | 0.7462 | 0.9980 | 0.6660 |
| yuanzhu__models__孔口 | 0.8551 | 0.8043 | 0.9200 | 0.4545 | 0.7419 | 0.6250 | 0.8214 | 0.5263 | 0.9075 | 0.8092 | 0.9089 | 0.4226 |
| yuanzhu__models__端面 | 0.9455 | 0.2051 | 0.9167 | 0.1842 | 0.9565 | 1.0000 | 0.9362 | 0.3111 | 0.9864 | 0.2321 | 0.9836 | 0.2553 |

## 各子类耗时与 YOLO 专属结构指标

| 子类 | AHL ms | YOLO ms | 更快方法 | YOLO mAP50(B) | YOLO mAP>=95% | YOLO IoU | YOLO IoU>=0.7 |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- |
| models__球面斜拍 | 13.3860 | 30.8311 | AHL | 0.8811 | FAIL | 0.8241 | PASS |
| qiusai__models__底面检测__端面检测 | 11.0852 | 31.6842 | AHL | 0.6682 | FAIL | 0.8965 | PASS |
| qiusai__models__球面俯拍 | 11.7827 | 33.9927 | AHL | 0.8958 | FAIL | 0.8955 | PASS |
| yuanzhu__models__内孔中 | 15.8533 | 31.9569 | AHL | 0.7486 | FAIL | 0.5623 | FAIL |
| yuanzhu__models__孔口 | 13.1665 | 35.0762 | AHL | 0.6170 | FAIL | 0.8552 | PASS |
| yuanzhu__models__端面 | 10.4143 | 36.0205 | AHL | 0.6353 | FAIL | 0.7921 | PASS |

## 阅读说明

- 若要直接比较 AHL 与 YOLO11，应优先看图像级指标：Accuracy、Precision、Recall、F1、AUC-ROC 和 AUC-PR。
- YOLO11 的 `mAP50(B)` 和 mask IoU 对分析 YOLO 本身很有价值，但它们在 AHL 中没有对应指标。
- `Inference ms` 需要谨慎解读，因为 AHL 的耗时可能只是特征级 forward 时间，而不是完整的从输入图像到输出分数的端到端耗时。

## 数据来源

- AHL 来源：`/ghome/huangjd/code/detected/summary/ahl_aligned_metrics_full.md`
- YOLO11 来源：`/ghome/huangjd/code/detected/summary/yolo11_aligned_metrics_full.md`
- 本报告：`/ghome/huangjd/code/detected/summary/ahl_vs_yolo11_aligned_metrics_comparison.md`
