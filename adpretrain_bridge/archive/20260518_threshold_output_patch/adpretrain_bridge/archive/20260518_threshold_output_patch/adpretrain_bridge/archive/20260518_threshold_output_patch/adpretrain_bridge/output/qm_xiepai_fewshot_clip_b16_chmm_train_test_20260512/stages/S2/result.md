# S2 Train/Test Metrics

Primary policy: `adpretrain_eval_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8269 | 0.7167 | 0.7679 | 0.8700 | 0.8949 | 0.7928 | 4.0388 |

## Threshold Results

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adpretrain_eval_best_f1 | 4.1597 | 0.8700 | 0.8269 | 0.7167 | 0.7679 |
| test_normal_p95_0 | 4.2571 | 0.8700 | 0.8542 | 0.6833 | 0.7593 |
| test_normal_p97_5 | 4.4416 | 0.8350 | 0.8857 | 0.5167 | 0.6526 |
| test_normal_p99_0 | 4.5990 | 0.7600 | 0.8750 | 0.2333 | 0.3684 |

## Scope

- No calibration samples are used or present.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.
- This is an evaluation metric, not a deployable threshold calibration mechanism.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
