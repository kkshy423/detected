# S5 Train/Test Metrics

Primary policy: `adpretrain_eval_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8679 | 0.7667 | 0.8142 | 0.8950 | 0.8940 | 0.8471 | 4.6821 |

## Threshold Results

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adpretrain_eval_best_f1 | 0.3536 | 0.8950 | 0.8679 | 0.7667 | 0.8142 |
| test_normal_p95_0 | 0.3535 | 0.8950 | 0.8679 | 0.7667 | 0.8142 |
| test_normal_p97_5 | 0.4111 | 0.8700 | 0.9048 | 0.6333 | 0.7451 |
| test_normal_p99_0 | 0.4612 | 0.8450 | 0.9394 | 0.5167 | 0.6667 |

## Scope

- No calibration samples are used or present.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.
- This is an evaluation metric, not a deployable threshold calibration mechanism.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
