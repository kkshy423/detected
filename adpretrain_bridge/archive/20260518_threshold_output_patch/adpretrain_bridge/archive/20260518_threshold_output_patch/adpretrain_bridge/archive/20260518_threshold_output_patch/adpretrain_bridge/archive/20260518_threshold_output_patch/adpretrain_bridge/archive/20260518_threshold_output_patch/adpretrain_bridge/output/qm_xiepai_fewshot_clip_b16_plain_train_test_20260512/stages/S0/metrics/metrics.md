# S0 Train/Test Metrics

Primary policy: `adpretrain_eval_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5254 | 0.5167 | 0.5210 | 0.7150 | 0.6831 | 0.5246 | 9.6924 |

## Threshold Results

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adpretrain_eval_best_f1 | 0.1161 | 0.7150 | 0.5254 | 0.5167 | 0.5210 |
| test_normal_p95_0 | 0.2063 | 0.7250 | 0.6316 | 0.2000 | 0.3038 |
| test_normal_p97_5 | 0.2205 | 0.7250 | 0.6923 | 0.1500 | 0.2466 |
| test_normal_p99_0 | 0.2342 | 0.7150 | 0.7143 | 0.0833 | 0.1493 |

## Scope

- No calibration samples are used or present.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores for parity with no-AHL evaluation.
- This is an evaluation metric, not a deployable threshold calibration mechanism.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
