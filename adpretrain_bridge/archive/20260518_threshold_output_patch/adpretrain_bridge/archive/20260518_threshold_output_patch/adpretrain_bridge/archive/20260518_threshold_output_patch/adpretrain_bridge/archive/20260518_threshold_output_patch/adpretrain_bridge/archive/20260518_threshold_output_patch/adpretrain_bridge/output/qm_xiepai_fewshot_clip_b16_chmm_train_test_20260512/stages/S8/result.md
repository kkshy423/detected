# S8 Train/Test Metrics

Primary policy: `adpretrain_eval_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8197 | 0.8333 | 0.8264 | 0.8950 | 0.9150 | 0.8338 | 5.8382 |

## Threshold Results

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adpretrain_eval_best_f1 | 1.3918 | 0.8950 | 0.8197 | 0.8333 | 0.8264 |
| test_normal_p95_0 | 1.5397 | 0.8950 | 0.8679 | 0.7667 | 0.8142 |
| test_normal_p97_5 | 1.7970 | 0.8350 | 0.8857 | 0.5167 | 0.6526 |
| test_normal_p99_0 | 1.9670 | 0.8000 | 0.9167 | 0.3667 | 0.5238 |

## Scope

- No calibration samples are used or present.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.
- This is an evaluation metric, not a deployable threshold calibration mechanism.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
