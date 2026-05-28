# S0 Metrics

Primary threshold policy: `normal_p97_5`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8667 | 0.2167 | 0.3467 | 0.7550 | 0.7888 | 0.6907 | 9.6401 |

## Thresholds

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal_p95_0 | -0.1008 | 0.7800 | 0.8636 | 0.3167 | 0.4634 |
| normal_p97_5 | 0.0674 | 0.7550 | 0.8667 | 0.2167 | 0.3467 |
| normal_p99_0 | 0.1683 | 0.7300 | 0.8750 | 0.1167 | 0.2059 |

## Scope

- Thresholds are selected only from calibration scores.
- Fixed test metrics use only `test_normal` and `test_anomaly` roles.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
