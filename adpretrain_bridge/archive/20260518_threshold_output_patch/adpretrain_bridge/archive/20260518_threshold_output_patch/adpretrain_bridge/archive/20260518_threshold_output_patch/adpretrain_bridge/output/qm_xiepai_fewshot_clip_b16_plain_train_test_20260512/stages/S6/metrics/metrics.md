# S6 Train/Test Metrics

Primary policy: `adpretrain_eval_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8824 | 0.7500 | 0.8108 | 0.8950 | 0.9280 | 0.8702 | 7.8726 |

## Threshold Results

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adpretrain_eval_best_f1 | 0.3597 | 0.8950 | 0.8824 | 0.7500 | 0.8108 |
| test_normal_p95_0 | 0.3559 | 0.8900 | 0.8654 | 0.7500 | 0.8036 |
| test_normal_p97_5 | 0.4128 | 0.8650 | 0.9024 | 0.6167 | 0.7327 |
| test_normal_p99_0 | 0.4655 | 0.8450 | 0.9394 | 0.5167 | 0.6667 |

## Scope

- No calibration samples are used or present.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.
- This is an evaluation metric, not a deployable threshold calibration mechanism.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
