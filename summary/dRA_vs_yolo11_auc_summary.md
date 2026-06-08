# DRA vs YOLO11 Baseline 汇总

## 结论速览

- 对齐子类数：**6**
- DRA 平均 AUC-ROC：**0.9355**；YOLO11 平均 AUC-ROC：**0.6596**；差值（DRA-YOLO11）：**0.2759**
- DRA 平均 AUC-PR：**0.9666**；YOLO11 平均 AUC-PR：**0.6182**；差值（DRA-YOLO11）：**0.3483**
- AUC-ROC 胜场：DRA=6，YOLO11=0，Tie=0
- AUC-PR 胜场：DRA=6，YOLO11=0，Tie=0

## 方法均值

| 方法 | 子类数 | 平均 AUC-ROC | 平均 AUC-PR | 备注 |
| --- | ---: | ---: | ---: | --- |
| DRA | 6 | 0.9355 | 0.9666 | 来自各子类 `train.log` 最终 AUC |
| YOLO11 baseline | 6 | 0.6596 | 0.6182 | 来自 `summary_last_epoch_and_auc.json` 的离线补算 AUC |

## 各子类 AUC 对比

| 子类 | DRA AUC-ROC | YOLO11 AUC-ROC | ΔROC (DRA-YOLO) | DRA AUC-PR | YOLO11 AUC-PR | ΔPR (DRA-YOLO) | YOLO11 mAP50(M) | YOLO11 mAP50-95(M) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| models__球面斜拍 | 0.9395 | 0.9361 | 0.0034 | 0.9609 | 0.8976 | 0.0633 | 0.8811 | 0.6262 |
| qiusai__models__底面检测__端面检测 | 0.8421 | 0.7380 | 0.1041 | 0.9453 | 0.7196 | 0.2257 | 0.6682 | 0.5345 |
| qiusai__models__球面俯拍 | 0.9304 | 0.5000 | 0.4304 | 0.9918 | 0.7507 | 0.2411 | 0.8958 | 0.7045 |
| yuanzhu__models__内孔中 | 0.9943 | 0.7423 | 0.2520 | 0.9980 | 0.6634 | 0.3346 | 0.7486 | 0.5766 |
| yuanzhu__models__孔口 | 0.9202 | 0.8092 | 0.1110 | 0.9197 | 0.4229 | 0.4968 | 0.6170 | 0.4329 |
| yuanzhu__models__端面 | 0.9864 | 0.2321 | 0.7543 | 0.9836 | 0.2553 | 0.7283 | 0.6353 | 0.3812 |

## 分项观察

- **models__球面斜拍**：AUC-ROC DRA 更优（Δ=0.0034），AUC-PR DRA 更优（Δ=0.0633）
- **qiusai__models__底面检测__端面检测**：AUC-ROC DRA 更优（Δ=0.1041），AUC-PR DRA 更优（Δ=0.2257）
- **qiusai__models__球面俯拍**：AUC-ROC DRA 更优（Δ=0.4304），AUC-PR DRA 更优（Δ=0.2411）
- **yuanzhu__models__内孔中**：AUC-ROC DRA 更优（Δ=0.2520），AUC-PR DRA 更优（Δ=0.3346）
- **yuanzhu__models__孔口**：AUC-ROC DRA 更优（Δ=0.1110），AUC-PR DRA 更优（Δ=0.4968）
- **yuanzhu__models__端面**：AUC-ROC DRA 更优（Δ=0.7543），AUC-PR DRA 更优（Δ=0.7283）

## 数据来源

- DRA 日志目录：`/ghome/huangjd/code/detected/DRA/experiment/xidun`
- YOLO baseline 汇总：`/ghome/huangjd/code/detected/yolo11_baselines/output/summary_last_epoch_and_auc.json`
- 本报告路径：`/ghome/huangjd/code/detected/summary/dRA_vs_yolo11_auc_summary.md`
