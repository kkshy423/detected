# S2 Metrics

Primary threshold policy: `normal_p97_5`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8462 | 0.7333 | 0.7857 | 0.8800 | 0.9023 | 0.8014 | 5.8732 |

## Thresholds

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal_p95_0 | 2.1918 | 0.8650 | 0.7797 | 0.7667 | 0.7731 |
| normal_p97_5 | 2.2640 | 0.8800 | 0.8462 | 0.7333 | 0.7857 |
| normal_p99_0 | 2.2799 | 0.8800 | 0.8462 | 0.7333 | 0.7857 |

## Scope

- Thresholds are selected only from calibration scores.
- Fixed test metrics use only `test_normal` and `test_anomaly` roles.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
