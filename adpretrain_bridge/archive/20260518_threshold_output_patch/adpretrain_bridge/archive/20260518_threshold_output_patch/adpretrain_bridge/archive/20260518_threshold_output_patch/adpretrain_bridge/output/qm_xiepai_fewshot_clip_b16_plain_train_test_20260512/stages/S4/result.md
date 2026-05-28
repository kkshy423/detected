# S4 Train/Test Metrics

Primary policy: `adpretrain_eval_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7879 | 0.8667 | 0.8254 | 0.8900 | 0.9279 | 0.8727 | 6.5100 |

## Threshold Results

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adpretrain_eval_best_f1 | 0.3068 | 0.8900 | 0.7879 | 0.8667 | 0.8254 |
| test_normal_p95_0 | 0.3345 | 0.8850 | 0.8627 | 0.7333 | 0.7928 |
| test_normal_p97_5 | 0.3850 | 0.8700 | 0.9048 | 0.6333 | 0.7451 |
| test_normal_p99_0 | 0.4500 | 0.8150 | 0.9259 | 0.4167 | 0.5747 |

## Scope

- No calibration samples are used or present.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.
- This is an evaluation metric, not a deployable threshold calibration mechanism.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
