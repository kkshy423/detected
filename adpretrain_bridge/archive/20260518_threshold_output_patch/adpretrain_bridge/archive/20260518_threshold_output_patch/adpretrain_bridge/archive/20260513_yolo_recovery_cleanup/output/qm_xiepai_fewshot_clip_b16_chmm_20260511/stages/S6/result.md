# S6 Metrics

Primary threshold policy: `calib_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8571 | 0.6000 | 0.7059 | 0.8500 | 0.8949 | 0.8060 | 7.8507 |

## Thresholds

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal_p95_0 | 0.7364 | 0.8700 | 0.7931 | 0.7667 | 0.7797 |
| normal_p97_5 | 0.7708 | 0.8750 | 0.8070 | 0.7667 | 0.7863 |
| normal_p99_0 | 0.8245 | 0.8800 | 0.8462 | 0.7333 | 0.7857 |
| calib_best_f1 | 0.9256 | 0.8500 | 0.8571 | 0.6000 | 0.7059 |
| calib_recall_priority | 0.6158 | 0.8200 | 0.6538 | 0.8500 | 0.7391 |
| calib_precision_priority | 1.1197 | 0.8000 | 0.8846 | 0.3833 | 0.5349 |

## Scope

- Thresholds are selected only from calibration scores.
- Fixed test metrics use only `test_normal` and `test_anomaly` roles.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
