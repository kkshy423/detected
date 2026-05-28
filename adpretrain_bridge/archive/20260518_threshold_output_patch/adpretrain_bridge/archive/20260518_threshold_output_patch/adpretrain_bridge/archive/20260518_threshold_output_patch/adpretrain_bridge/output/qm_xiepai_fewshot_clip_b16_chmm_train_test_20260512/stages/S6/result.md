# S6 Train/Test Metrics

Primary policy: `adpretrain_eval_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8800 | 0.7333 | 0.8000 | 0.8900 | 0.9044 | 0.8257 | 5.8870 |

## Threshold Results

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adpretrain_eval_best_f1 | 4.9538 | 0.8900 | 0.8800 | 0.7333 | 0.8000 |
| test_normal_p95_0 | 4.8802 | 0.8850 | 0.8627 | 0.7333 | 0.7928 |
| test_normal_p97_5 | 5.1087 | 0.8400 | 0.8889 | 0.5333 | 0.6667 |
| test_normal_p99_0 | 5.3023 | 0.7900 | 0.9091 | 0.3333 | 0.4878 |

## Scope

- No calibration samples are used or present.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.
- This is an evaluation metric, not a deployable threshold calibration mechanism.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
