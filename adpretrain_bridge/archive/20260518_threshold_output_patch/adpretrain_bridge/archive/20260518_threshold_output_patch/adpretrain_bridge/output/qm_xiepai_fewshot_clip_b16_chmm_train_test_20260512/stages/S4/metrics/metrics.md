# S4 Train/Test Metrics

Primary policy: `adpretrain_eval_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8246 | 0.7833 | 0.8034 | 0.8850 | 0.8990 | 0.8124 | 6.6404 |

## Threshold Results

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adpretrain_eval_best_f1 | 0.2883 | 0.8850 | 0.8246 | 0.7833 | 0.8034 |
| test_normal_p95_0 | 0.3716 | 0.8750 | 0.8571 | 0.7000 | 0.7706 |
| test_normal_p97_5 | 0.5992 | 0.8250 | 0.8788 | 0.4833 | 0.6237 |
| test_normal_p99_0 | 0.7198 | 0.7900 | 0.9091 | 0.3333 | 0.4878 |

## Scope

- No calibration samples are used or present.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.
- This is an evaluation metric, not a deployable threshold calibration mechanism.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
