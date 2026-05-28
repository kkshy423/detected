# S8 Train/Test Metrics

Primary policy: `adpretrain_eval_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8136 | 0.8000 | 0.8067 | 0.8850 | 0.9224 | 0.8654 | 6.7572 |

## Threshold Results

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adpretrain_eval_best_f1 | 0.3764 | 0.8850 | 0.8136 | 0.8000 | 0.8067 |
| test_normal_p95_0 | 0.3936 | 0.8850 | 0.8627 | 0.7333 | 0.7928 |
| test_normal_p97_5 | 0.4712 | 0.8650 | 0.9024 | 0.6167 | 0.7327 |
| test_normal_p99_0 | 0.5233 | 0.8300 | 0.9333 | 0.4667 | 0.6222 |

## Scope

- No calibration samples are used or present.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.
- This is an evaluation metric, not a deployable threshold calibration mechanism.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
