# YOLO26s-cls S5 Recovered Eval

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8966 | 0.8667 | 0.8814 | 0.9300 | 0.9415 | 0.9112 | 36.0464 |

- Weights: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3/stages/S5/ultralytics/train/weights/best.pt`
- Scores: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3_recovered_eval/stages/S5/metrics/scores.csv`
- Evaluation policy: YOLO defect probability, fixed-test best-F1 threshold.
