# S7 Train/Test Metrics

Primary policy: `adpretrain_eval_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8393 | 0.7833 | 0.8103 | 0.8900 | 0.9187 | 0.8625 | 6.0340 |

## Threshold Results

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adpretrain_eval_best_f1 | 0.3464 | 0.8900 | 0.8393 | 0.7833 | 0.8103 |
| test_normal_p95_0 | 0.3617 | 0.8850 | 0.8627 | 0.7333 | 0.7928 |
| test_normal_p97_5 | 0.4394 | 0.8650 | 0.9024 | 0.6167 | 0.7327 |
| test_normal_p99_0 | 0.4794 | 0.8350 | 0.9355 | 0.4833 | 0.6374 |

## Scope

- No calibration samples are used or present.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.
- This is an evaluation metric, not a deployable threshold calibration mechanism.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
