# 20260530 Stage Recall-Constrained Best-F1 Policy

Eval-only summary from existing score files. Thresholds are selected from calibration/val scores only; test labels are used only for reporting.

Policy:

- S0-S2: choose calibration best-F1 among thresholds with calibration recall >= 0.90.
- S3-S6: choose calibration best-F1 among thresholds with calibration recall >= 0.85.
- S7-S8: choose unconstrained calibration best-F1.

## AHL-DINO

| Stage | Policy | Threshold | Calib R | Calib F1 | Calib FPR | Test P | Test R | Test F1 | Test FP | Test FN | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | calib_best_f1_with_recall_ge_0.90 | -0.6530 | 0.9184 | 0.5660 | 0.6500 | 0.4026 | 0.8857 | 0.5536 | 92 | 8 | 0.8473 | 0.7653 |
| S1 | calib_best_f1_with_recall_ge_0.90 | 0.1634 | 0.9388 | 0.6093 | 0.5600 | 0.5943 | 0.9000 | 0.7159 | 43 | 7 | 0.9407 | 0.8926 |
| S2 | calib_best_f1_with_recall_ge_0.90 | 0.1809 | 0.9388 | 0.5823 | 0.6300 | 0.3681 | 0.9571 | 0.5317 | 115 | 3 | 0.8650 | 0.8122 |
| S3 | calib_best_f1_with_recall_ge_0.85 | 0.2899 | 0.8571 | 0.6462 | 0.3900 | 0.6598 | 0.9143 | 0.7665 | 33 | 6 | 0.9308 | 0.8905 |
| S4 | calib_best_f1_with_recall_ge_0.85 | 0.4144 | 0.8776 | 0.6014 | 0.5100 | 0.6436 | 0.9286 | 0.7602 | 36 | 5 | 0.9392 | 0.9041 |
| S5 | calib_best_f1_with_recall_ge_0.85 | 0.4376 | 0.8571 | 0.6269 | 0.4300 | 0.7126 | 0.8857 | 0.7898 | 25 | 8 | 0.9440 | 0.9059 |
| S6 | calib_best_f1_with_recall_ge_0.85 | 0.5001 | 0.9184 | 0.6250 | 0.5000 | 0.6875 | 0.9429 | 0.7952 | 30 | 4 | 0.9560 | 0.9191 |
| S7 | calib_best_f1 | 0.5805 | 0.6735 | 0.7416 | 0.0700 | 0.9592 | 0.6714 | 0.7899 | 2 | 23 | 0.9515 | 0.9118 |
| S8 | calib_best_f1 | 0.4778 | 0.7551 | 0.7184 | 0.1700 | 0.8485 | 0.8000 | 0.8235 | 10 | 14 | 0.9362 | 0.8881 |

## ADP-only-DINO

| Stage | Policy | Threshold | Calib R | Calib F1 | Calib FPR | Test P | Test R | Test F1 | Test FP | Test FN | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | calib_best_f1_with_recall_ge_0.90 | 1.3744 | 0.9184 | 0.6522 | 0.4400 | 0.6389 | 0.9857 | 0.7753 | 39 | 1 | 0.9690 | 0.9151 |
| S1 | calib_best_f1_with_recall_ge_0.90 | 1.3744 | 0.9184 | 0.6522 | 0.4400 | 0.6389 | 0.9857 | 0.7753 | 39 | 1 | 0.9690 | 0.9151 |
| S2 | calib_best_f1_with_recall_ge_0.90 | 1.3744 | 0.9184 | 0.6522 | 0.4400 | 0.6389 | 0.9857 | 0.7753 | 39 | 1 | 0.9690 | 0.9151 |
| S3 | calib_best_f1_with_recall_ge_0.85 | 1.3926 | 0.8980 | 0.6769 | 0.3700 | 0.6635 | 0.9857 | 0.7931 | 35 | 1 | 0.9690 | 0.9151 |
| S4 | calib_best_f1_with_recall_ge_0.85 | 1.3926 | 0.8980 | 0.6769 | 0.3700 | 0.6635 | 0.9857 | 0.7931 | 35 | 1 | 0.9690 | 0.9151 |
| S5 | calib_best_f1_with_recall_ge_0.85 | 1.3926 | 0.8980 | 0.6769 | 0.3700 | 0.6635 | 0.9857 | 0.7931 | 35 | 1 | 0.9690 | 0.9151 |
| S6 | calib_best_f1_with_recall_ge_0.85 | 1.3926 | 0.8980 | 0.6769 | 0.3700 | 0.6635 | 0.9857 | 0.7931 | 35 | 1 | 0.9690 | 0.9151 |
| S7 | calib_best_f1 | 1.7079 | 0.7347 | 0.7200 | 0.1500 | 0.8429 | 0.8429 | 0.8429 | 11 | 11 | 0.9690 | 0.9151 |
| S8 | calib_best_f1 | 1.7079 | 0.7347 | 0.7200 | 0.1500 | 0.8429 | 0.8429 | 0.8429 | 11 | 11 | 0.9690 | 0.9151 |

## YOLO26s-eval

| Stage | Policy | Threshold | Calib R | Calib F1 | Calib FPR | Test P | Test R | Test F1 | Test FP | Test FN | AUC-ROC | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | calib_best_f1_with_recall_ge_0.90 | 0.0000 | 1.0000 | 0.4949 | 1.0000 | 0.2811 | 1.0000 | 0.4389 | 179 | 0 | 0.8906 | 0.8139 |
| S2 | calib_best_f1_with_recall_ge_0.90 | 0.0020 | 0.9184 | 0.5696 | 0.6400 | 0.4397 | 0.8857 | 0.5877 | 79 | 8 | 0.8552 | 0.7987 |
| S3 | calib_best_f1_with_recall_ge_0.85 | 0.0022 | 0.8571 | 0.5490 | 0.6200 | 0.3367 | 0.9429 | 0.4962 | 130 | 4 | 0.8744 | 0.8566 |
| S4 | calib_best_f1_with_recall_ge_0.85 | 0.0007 | 0.8571 | 0.5714 | 0.5600 | 0.5200 | 0.9286 | 0.6667 | 60 | 5 | 0.9237 | 0.9023 |
| S5 | calib_best_f1_with_recall_ge_0.85 | 0.0017 | 0.8776 | 0.5850 | 0.5500 | 0.4621 | 0.9571 | 0.6233 | 78 | 3 | 0.9370 | 0.9169 |
| S6 | calib_best_f1_with_recall_ge_0.85 | 0.0226 | 0.9388 | 0.7731 | 0.2400 | 0.7727 | 0.9714 | 0.8608 | 20 | 2 | 0.9817 | 0.9697 |
| S7 | calib_best_f1 | 0.2615 | 0.7959 | 0.8387 | 0.0500 | 0.8955 | 0.8571 | 0.8759 | 7 | 10 | 0.9674 | 0.9464 |
| S8 | calib_best_f1 | 0.4312 | 0.8163 | 0.8511 | 0.0500 | 0.8986 | 0.8857 | 0.8921 | 7 | 8 | 0.9790 | 0.9638 |

## Short Diagnosis

- This table tests the requested threshold policy only; it does not compare against test-best thresholds.
- If `constraint_satisfied` is false in the CSV, the requested calibration recall floor was infeasible for that score set.
- S7-S8 intentionally allow lower calibration recall if that is where calibration F1 is maximized.

## Unavailable Inputs

| Method | Stage | Missing file |
|---|---|---|
| YOLO26s-eval | S0 | `output/20260529_yolo26s_cls_fixed_180_70_val49_eval_only/stages/S0/metrics/scores.csv` |
