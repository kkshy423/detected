# YOLO26s-cls S1 Recovered Eval

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Infer ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8421 | 0.5333 | 0.6531 | 0.8300 | 0.7298 | 0.7017 | 37.3701 |

- Weights: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3/stages/S1/ultralytics/train/weights/best.pt`
- Scores: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3_recovered_eval/stages/S1/metrics/scores.csv`
- Evaluation policy: YOLO defect probability, fixed-test best-F1 threshold.
