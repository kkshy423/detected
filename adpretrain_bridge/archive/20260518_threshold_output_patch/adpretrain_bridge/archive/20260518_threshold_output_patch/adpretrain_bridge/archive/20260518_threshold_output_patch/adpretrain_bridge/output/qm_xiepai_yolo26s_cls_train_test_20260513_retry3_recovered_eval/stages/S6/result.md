# YOLO26s-cls S6 Recovered Eval

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8750 | 0.8167 | 0.8448 | 0.9100 | 0.9121 | 0.8961 | 31.4283 |

- Weights: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3/stages/S6/ultralytics/train/weights/best.pt`
- Scores: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3_recovered_eval/stages/S6/metrics/scores.csv`
- Evaluation policy: YOLO defect probability, fixed-test best-F1 threshold.
