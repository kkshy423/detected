# YOLO11 Baseline 指标汇总

- 说明：`precision_M` / `recall_M` / `mAP` 来自 YOLO `results.csv` 的最后一轮验证指标；`offline_auc_roc` / `offline_auc_pr` 来自离线图像级异常分数补算。
- 注意：这两组指标**不是完全同一评测协议**。前者偏检测/分割验证，后者偏图像级异常判别。

## 均值

| 子类数 | 平均 Precision(M) | 平均 Recall(M) | 平均 mAP50(M) | 平均 mAP50-95(M) | 平均 AUC-ROC | 平均 AUC-PR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.7978 | 0.6597 | 0.7410 | 0.5427 | 0.6596 | 0.6182 |

## 各子类

| 子类 | Precision(M) | Recall(M) | mAP50(M) | mAP50-95(M) | AUC-ROC | AUC-PR | Last Epoch | 测试正常 | 测试异常 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| models__球面斜拍 | 0.9100 | 0.7588 | 0.8811 | 0.6262 | 0.9361 | 0.8976 | 200 | 140 | 40 |
| qiusai__models__底面检测__端面检测 | 0.8900 | 0.3947 | 0.6682 | 0.5345 | 0.7380 | 0.7196 | 144 | 74 | 41 |
| qiusai__models__球面俯拍 | 0.6595 | 1.0000 | 0.8958 | 0.7045 | 0.5000 | 0.7507 | 186 | 19 | 38 |
| yuanzhu__models__内孔中 | 0.7564 | 0.7692 | 0.7486 | 0.5766 | 0.7423 | 0.6634 | 200 | 20 | 13 |
| yuanzhu__models__孔口 | 0.5707 | 0.7500 | 0.6170 | 0.4329 | 0.8092 | 0.4229 | 200 | 38 | 8 |
| yuanzhu__models__端面 | 1.0000 | 0.2853 | 0.6353 | 0.3812 | 0.2321 | 0.2553 | 200 | 32 | 7 |

## 输出文件

- Markdown: `/ghome/huangjd/code/detected/summary/yolo11_metrics_summary.md`
- CSV: `/ghome/huangjd/code/detected/summary/yolo11_metrics_summary.csv`
