# 20260530 DINO Late Threshold Diagnostics

This report is eval-only. Every probed threshold is selected from calibration scores only. Held-out test labels are used only to report resulting metrics.

## A. Is AHL-DINO late recall mainly a threshold issue or a ranking issue?

| Stage | Test AUC-ROC | Test AUC-PR | Current threshold | Current R | Current FP | Current FN | recall90_fpr20 threshold | recall90_fpr20 R | recall90_fpr20 FP | recall90_fpr20 FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S7 | 0.9515 | 0.9118 | 0.5805 | 0.6714 | 2 | 23 | 0.5104 | 0.8286 | 10 | 12 |
| S8 | 0.9362 | 0.8881 | 0.5400 | 0.5857 | 3 | 29 | 0.4778 | 0.8000 | 10 | 14 |

AHL-DINO retains strong test ranking metrics in both stages. Under strict FPR caps, however, recall cannot be fully restored without accepting more false positives. The low late-stage recall is therefore a threshold operating-point issue under the current low-FPR target, not evidence that the ranking has collapsed.

## B. Is there an obvious calib-defect versus test-defect score shift?

| Stage | Calib defect p25 | Calib defect p50 | Calib defect p75 | Test defect p25 | Test defect p50 | Test defect p75 | Test median - calib median |
|---|---:|---:|---:|---:|---:|---:|---:|
| S7 | 0.4735 | 0.6497 | 0.8495 | 0.5623 | 0.7014 | 0.8101 | 0.0517 |
| S8 | 0.4778 | 0.5491 | 0.7154 | 0.4906 | 0.5946 | 0.6771 | 0.0455 |

The defect-score distributions should be read together with the table above. A negative median delta means test defects score lower than calibration defects; a positive value means the opposite. This quantifies distribution shift without using test labels to select a threshold.

## C. Which calibration-derived threshold is more suitable for late AHL-DINO?

| Stage | Policy | Threshold | Calib P | Calib R | Calib F1 | Calib FPR | Test P | Test R | Test F1 | Test FP | Test FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S7 | current | 0.5805 | 0.8250 | 0.6735 | 0.7416 | 0.0700 | 0.9592 | 0.6714 | 0.7899 | 2 | 23 |
| S7 | val_best_f1 | 0.5805 | 0.8250 | 0.6735 | 0.7416 | 0.0700 | 0.9592 | 0.6714 | 0.7899 | 2 | 23 |
| S7 | val_best_f1_fpr10 | 0.5805 | 0.8250 | 0.6735 | 0.7416 | 0.0700 | 0.9592 | 0.6714 | 0.7899 | 2 | 23 |
| S7 | recall90_fpr20 | 0.5104 | 0.6923 | 0.7347 | 0.7129 | 0.1600 | 0.8529 | 0.8286 | 0.8406 | 10 | 12 |
| S7 | recall85_fpr10 | 0.5805 | 0.8250 | 0.6735 | 0.7416 | 0.0700 | 0.9592 | 0.6714 | 0.7899 | 2 | 23 |
| S7 | recall80_fpr10 | 0.5805 | 0.8250 | 0.6735 | 0.7416 | 0.0700 | 0.9592 | 0.6714 | 0.7899 | 2 | 23 |
| S7 | normal_p95 | 0.5928 | 0.8611 | 0.6327 | 0.7294 | 0.0500 | 0.9556 | 0.6143 | 0.7478 | 2 | 27 |
| S7 | normal_p90 | 0.5634 | 0.7674 | 0.6735 | 0.7174 | 0.1000 | 0.9286 | 0.7429 | 0.8254 | 4 | 18 |
| S8 | current | 0.5400 | 0.7297 | 0.5510 | 0.6279 | 0.1000 | 0.9318 | 0.5857 | 0.7193 | 3 | 29 |
| S8 | val_best_f1 | 0.4778 | 0.6852 | 0.7551 | 0.7184 | 0.1700 | 0.8485 | 0.8000 | 0.8235 | 10 | 14 |
| S8 | val_best_f1_fpr10 | 0.5400 | 0.7297 | 0.5510 | 0.6279 | 0.1000 | 0.9318 | 0.5857 | 0.7193 | 3 | 29 |
| S8 | recall90_fpr20 | 0.4778 | 0.6852 | 0.7551 | 0.7184 | 0.1700 | 0.8485 | 0.8000 | 0.8235 | 10 | 14 |
| S8 | recall85_fpr10 | 0.5400 | 0.7297 | 0.5510 | 0.6279 | 0.1000 | 0.9318 | 0.5857 | 0.7193 | 3 | 29 |
| S8 | recall80_fpr10 | 0.5400 | 0.7297 | 0.5510 | 0.6279 | 0.1000 | 0.9318 | 0.5857 | 0.7193 | 3 | 29 |
| S8 | normal_p95 | 0.5789 | 0.8214 | 0.4694 | 0.5974 | 0.0500 | 0.9487 | 0.5286 | 0.6789 | 2 | 33 |
| S8 | normal_p90 | 0.5389 | 0.7297 | 0.5510 | 0.6279 | 0.1000 | 0.9318 | 0.5857 | 0.7193 | 3 | 29 |

No single threshold is universally preferable without a production cost target. For a late-stage balanced operating point under a strict calibration false-positive cap, keep `val_best_f1_fpr10` / `current`. For a higher-recall production mode that accepts more false alarms, compare `recall90_fpr20` directly against the current policy. `normal_p90` and `normal_p95` are included as normal-only calibration references.

## Notes

- `current` is read from each method's existing `metrics.json` primary threshold.
- `val_best_f1_fpr10` maximizes calibration F1 among thresholds with calibration FPR <= 0.10.
- `recallXX_fprYY` uses calibration labels only and keeps the stated FPR cap. If the recall floor is infeasible, it reports the highest calibration recall reachable under the cap.
- `fn_score_margin.csv` uses the current threshold. `rank_percentile` is the FN score percentile within all test scores for the same method and stage.
- No test-label-selected threshold is exported as a production candidate.
