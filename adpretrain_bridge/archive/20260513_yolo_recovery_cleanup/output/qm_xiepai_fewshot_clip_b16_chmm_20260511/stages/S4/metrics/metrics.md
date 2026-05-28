# S4 Metrics

Primary threshold policy: `calib_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7222 | 0.8667 | 0.7879 | 0.8600 | 0.9345 | 0.8534 | 10.3424 |

## Thresholds

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal_p95_0 | 0.4599 | 0.8850 | 0.7846 | 0.8500 | 0.8160 |
| normal_p97_5 | 0.5170 | 0.9050 | 0.8596 | 0.8167 | 0.8376 |
| normal_p99_0 | 0.7811 | 0.8550 | 0.8974 | 0.5833 | 0.7071 |
| calib_best_f1 | 0.4167 | 0.8600 | 0.7222 | 0.8667 | 0.7879 |
| calib_recall_priority | 0.4167 | 0.8600 | 0.7222 | 0.8667 | 0.7879 |
| calib_precision_priority | 0.4283 | 0.8650 | 0.7324 | 0.8667 | 0.7939 |

## Scope

- Thresholds are selected only from calibration scores.
- Fixed test metrics use only `test_normal` and `test_anomaly` roles.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
