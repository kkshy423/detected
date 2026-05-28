# S0 Train/Test Metrics

Primary policy: `adpretrain_eval_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3000 | 1.0000 | 0.4615 | 0.3000 | 0.2831 | 0.2227 | 5.9078 |

## Threshold Results

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adpretrain_eval_best_f1 | -1.3686 | 0.3000 | 0.3000 | 1.0000 | 0.4615 |
| test_normal_p95_0 | 0.2749 | 0.6750 | 0.2222 | 0.0333 | 0.0580 |
| test_normal_p97_5 | 0.3775 | 0.6850 | 0.2000 | 0.0167 | 0.0308 |
| test_normal_p99_0 | 0.6085 | 0.6900 | 0.0000 | 0.0000 | 0.0000 |

## Scope

- No calibration samples are used or present.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.
- This is an evaluation metric, not a deployable threshold calibration mechanism.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
