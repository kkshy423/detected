# YOLO26s-cls S2 Recovered Eval

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.4828 | 0.7000 | 0.5714 | 0.6850 | 0.7074 | 0.5651 | 35.6445 |

- Weights: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3/stages/S2/ultralytics/train/weights/best.pt`
- Scores: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3_recovered_eval/stages/S2/metrics/scores.csv`
- Evaluation policy: YOLO defect probability, fixed-test best-F1 threshold.
