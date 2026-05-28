# S1 Train/Test Metrics

Primary policy: `adpretrain_eval_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6724 | 0.6500 | 0.6610 | 0.8000 | 0.7846 | 0.6956 | 6.3768 |

## Threshold Results

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adpretrain_eval_best_f1 | 0.3127 | 0.8000 | 0.6724 | 0.6500 | 0.6610 |
| test_normal_p95_0 | 0.3533 | 0.8200 | 0.8158 | 0.5167 | 0.6327 |
| test_normal_p97_5 | 0.3953 | 0.8200 | 0.8750 | 0.4667 | 0.6087 |
| test_normal_p99_0 | 0.4566 | 0.7900 | 0.9091 | 0.3333 | 0.4878 |

## Scope

- No calibration samples are used or present.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.
- This is an evaluation metric, not a deployable threshold calibration mechanism.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
