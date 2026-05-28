# S8 Metrics

Primary threshold policy: `calib_best_f1`

| Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7167 | 0.7167 | 0.7167 | 0.8300 | 0.8773 | 0.7751 | 5.8067 |

## Thresholds

| Policy | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal_p95_0 | 1.5722 | 0.8500 | 0.7679 | 0.7167 | 0.7414 |
| normal_p97_5 | 1.6808 | 0.8350 | 0.8140 | 0.5833 | 0.6796 |
| normal_p99_0 | 1.7048 | 0.8400 | 0.8333 | 0.5833 | 0.6863 |
| calib_best_f1 | 1.5369 | 0.8300 | 0.7167 | 0.7167 | 0.7167 |
| calib_recall_priority | 1.1787 | 0.3500 | 0.3158 | 1.0000 | 0.4800 |
| calib_precision_priority | 1.9506 | 0.7850 | 0.9474 | 0.3000 | 0.4557 |

## Scope

- Thresholds are selected only from calibration scores.
- Fixed test metrics use only `test_normal` and `test_anomaly` roles.
- Inference timing: feature-level AHL forward time; excludes CLIP extraction, CHMM affine, and disk I/O
