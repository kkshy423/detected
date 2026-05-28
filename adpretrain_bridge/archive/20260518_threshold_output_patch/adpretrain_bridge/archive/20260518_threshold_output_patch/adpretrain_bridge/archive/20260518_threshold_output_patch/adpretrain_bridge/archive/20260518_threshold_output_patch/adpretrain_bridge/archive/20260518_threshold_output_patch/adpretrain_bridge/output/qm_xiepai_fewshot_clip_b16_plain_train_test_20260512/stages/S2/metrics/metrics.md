# S2 Train/Test Metrics

Primary policy: `adpretrain_eval_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8400 | 0.7000 | 0.7636 | 0.8700 | 0.8736 | 0.7732 | 5.9608 |

## Threshold Results

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adpretrain_eval_best_f1 | 0.3431 | 0.8700 | 0.8400 | 0.7000 | 0.7636 |
| test_normal_p95_0 | 0.3541 | 0.8650 | 0.8511 | 0.6667 | 0.7477 |
| test_normal_p97_5 | 0.4186 | 0.8100 | 0.8667 | 0.4333 | 0.5778 |
| test_normal_p99_0 | 0.4771 | 0.7850 | 0.9048 | 0.3167 | 0.4691 |

## Scope

- No calibration samples are used or present.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.
- This is an evaluation metric, not a deployable threshold calibration mechanism.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
