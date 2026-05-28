# YOLO26s-cls S3 Recovered Eval

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.6809 | 0.5333 | 0.5981 | 0.7850 | 0.7264 | 0.6666 | 38.0783 |

- Weights: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3/stages/S3/ultralytics/train/weights/best.pt`
- Scores: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3_recovered_eval/stages/S3/metrics/scores.csv`
- Evaluation policy: YOLO defect probability, fixed-test best-F1 threshold.
