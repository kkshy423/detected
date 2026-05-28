# 球面斜拍主线实验结果表

更新时间：2026-05-15

说明：Precision / Recall / F1 当前均为 fixed-test best-F1 口径，用于离线公平比较；不是生产部署阈值。固定测试集为 140 normal + 60 anomaly。

## 1. 主线方法总览

| 方法 | 类型 | 覆盖阶段 | 代表 Precision | 代表 Recall | 代表 F1 | 代表 AUC-ROC | 代表 AUC-PR | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ADPretrain-only official fixed-test | 不训练 / 不微调 | S0-S8 共用同一次 fixed-test 结果 | 0.7714 | 0.9000 | 0.8308 | 0.9294 | 0.8246 | 强 baseline，冷启动稳定 |
| AHL plain no-CHMM | ADPretrain CLIP-B16 feature -> AHL | S0/S2/S4/S6/S8 | S4: 0.7879 | S4: 0.8667 | S4: 0.8254 | S4: 0.9279 | S4: 0.8727 | 当前 few-shot 主线 |
| AHL + CHMM | AHL ablation | S0/S2/S4/S6/S8 | S8: 0.8197 | S8: 0.8333 | S8: 0.8264 | S8: 0.9150 | S8: 0.8338 | 不稳定，不作为默认主线 |
| YOLO26s-cls recovered eval | 全监督分类 baseline | S1-S8，S0 skipped | S8: 0.9000 | S8: 0.9000 | S8: 0.9000 | S8: 0.9651 | S8: 0.9448 | S5 后明显变强，S8 达到切换门槛 |

## 2. ADPretrain-only 结果

| 评估集 | 阶段 | Train N/A | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-test 140/60 | S0-S8 repeated | 50/0 -> 560/139 | 0.7714 | 0.9000 | 0.8308 | 0.8900 | 0.9294 | 0.8246 | 28.2328 |
| full-original 140/199 | full_original | ref 560/0 | 0.8584 | 0.9447 | 0.8995 | 0.8761 | 0.9309 | 0.9366 | 25.5392 |

注：full-original 使用 199 张原始 test defect，和 fixed-test 60 张 anomaly 不是同一个测试集；它用于判断 fixed-test 是否更难。

## 3. AHL plain no-CHMM 阶段曲线

| Stage | Train N/A | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | 50/0 | 0.5254 | 0.5167 | 0.5210 | 0.7150 | 0.6831 | 0.5246 | 9.6924 |
| S2 | 150/3 | 0.8400 | 0.7000 | 0.7636 | 0.8700 | 0.8736 | 0.7732 | 5.9608 |
| S4 | 300/10 | 0.7879 | 0.8667 | 0.8254 | 0.8900 | 0.9279 | 0.8727 | 6.5100 |
| S6 | 500/40 | 0.8824 | 0.7500 | 0.8108 | 0.8950 | 0.9280 | 0.8702 | 7.8726 |
| S8 | 560/139 | 0.8136 | 0.8000 | 0.8067 | 0.8850 | 0.9224 | 0.8654 | 6.7572 |

## 4. AHL + CHMM ablation 阶段曲线

| Stage | Train N/A | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | 50/0 | 0.3000 | 1.0000 | 0.4615 | 0.3000 | 0.2831 | 0.2227 | 5.9078 |
| S2 | 150/3 | 0.8269 | 0.7167 | 0.7679 | 0.8700 | 0.8949 | 0.7928 | 4.0388 |
| S4 | 300/10 | 0.8246 | 0.7833 | 0.8034 | 0.8850 | 0.8990 | 0.8124 | 6.6404 |
| S6 | 500/40 | 0.8800 | 0.7333 | 0.8000 | 0.8900 | 0.9044 | 0.8257 | 5.8870 |
| S8 | 560/139 | 0.8197 | 0.8333 | 0.8264 | 0.8950 | 0.9150 | 0.8338 | 5.8382 |

## 5. YOLO26s-cls recovered eval 阶段曲线

| Stage | Train N/A | Status | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | 50/0 | skipped | - | - | - | - | - | - | - |
| S1 | 100/1 | ok | 0.8421 | 0.5333 | 0.6531 | 0.8300 | 0.7298 | 0.7017 | 37.3701 |
| S2 | 150/3 | ok | 0.4828 | 0.7000 | 0.5714 | 0.6850 | 0.7074 | 0.5651 | 35.6445 |
| S3 | 200/5 | ok | 0.6809 | 0.5333 | 0.5981 | 0.7850 | 0.7264 | 0.6666 | 38.0783 |
| S4 | 300/10 | ok | 0.6324 | 0.7167 | 0.6719 | 0.7900 | 0.8279 | 0.7479 | 32.8388 |
| S5 | 400/20 | ok | 0.8966 | 0.8667 | 0.8814 | 0.9300 | 0.9415 | 0.9112 | 36.0464 |
| S6 | 500/40 | ok | 0.8750 | 0.8167 | 0.8448 | 0.9100 | 0.9121 | 0.8961 | 31.4283 |
| S7 | 560/80 | ok | 0.9123 | 0.8667 | 0.8889 | 0.9350 | 0.9508 | 0.9236 | 31.1983 |
| S8 | 560/139 | ok | 0.9000 | 0.9000 | 0.9000 | 0.9400 | 0.9651 | 0.9448 | 34.7447 |

## 6. 同阶段横向对比

| Stage | Train N/A | ADPretrain-only F1 | AHL plain F1 | AHL+CHMM F1 | YOLO F1 | AHL plain AUC-PR | AHL+CHMM AUC-PR | YOLO AUC-PR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | 50/0 | 0.8308 | 0.5210 | 0.4615 | - | 0.5246 | 0.2227 | - |
| S2 | 150/3 | 0.8308 | 0.7636 | 0.7679 | 0.5714 | 0.7732 | 0.7928 | 0.5651 |
| S4 | 300/10 | 0.8308 | 0.8254 | 0.8034 | 0.6719 | 0.8727 | 0.8124 | 0.7479 |
| S6 | 500/40 | 0.8308 | 0.8108 | 0.8000 | 0.8448 | 0.8702 | 0.8257 | 0.8961 |
| S8 | 560/139 | 0.8308 | 0.8067 | 0.8264 | 0.9000 | 0.8654 | 0.8338 | 0.9448 |

## 7. 阶段决策表

| 阶段 | 数据状态 Train N/A | 推荐方法 | 依据 / 风险 | 决策 |
| --- | ---: | --- | --- | --- |
| S0 | 50/0 | ADPretrain-only 或 AHL plain | YOLO 无异常样本不能训练；AHL 仅适合初筛 | 不可直接上线，需人工复核 |
| S1-S4 | 100/1 -> 300/10 | AHL plain no-CHMM | YOLO 不稳定，S2/S4 明显弱于 AHL；当前 best-F1 不是生产阈值 | S4 可作为 few-shot 过渡候选 |
| S5 | 400/20 | YOLO26s-cls 进入候选 | YOLO F1=0.8814，首次明显超过 AHL/ADPretrain-only；需多 seed 验证 | 可启动全监督候选验证 |
| S6-S7 | 500/40 -> 560/80 | YOLO26s-cls + AHL 对照 | YOLO S6 有波动，S7 接近切换门槛 | 建议做稳定性和阈值校准 |
| S8 | 560/139 | YOLO26s-cls | YOLO P/R/F1=0.9000/0.9000/0.9000，离线达到切换门槛 | 可作为全监督切换点，但需生产阈值验证 |

## 8. 文件索引

- CSV 汇总：`/ghome/huangjd/code/detected/adpretrain_bridge/qm_xiepai_main_results_tables_20260515.csv`
- ADPretrain-only fixed-test：`output/qm_xiepai_adpretrain_only_clip_b16_official_train_test_20260512`
- ADPretrain-only full-original：`output/qm_xiepai_adpretrain_only_clip_b16_official_full_original_20260513`
- AHL plain：`output/qm_xiepai_fewshot_clip_b16_plain_train_test_20260512`
- AHL + CHMM：`output/qm_xiepai_fewshot_clip_b16_chmm_train_test_20260512`
- YOLO recovered eval：`output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3_recovered_eval`