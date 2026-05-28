# S3 Train/Test Metrics

Primary policy: `adpretrain_eval_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8704 | 0.7833 | 0.8246 | 0.9000 | 0.9194 | 0.8270 | 6.3975 |

## Threshold Results

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adpretrain_eval_best_f1 | 0.3651 | 0.9000 | 0.8704 | 0.7833 | 0.8246 |
| test_normal_p95_0 | 0.3653 | 0.8950 | 0.8679 | 0.7667 | 0.8142 |
| test_normal_p97_5 | 0.4230 | 0.8450 | 0.8919 | 0.5500 | 0.6804 |
| test_normal_p99_0 | 0.5028 | 0.7700 | 0.8889 | 0.2667 | 0.4103 |

## Scope

- No calibration samples are used or present.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.
- This is an evaluation metric, not a deployable threshold calibration mechanism.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
