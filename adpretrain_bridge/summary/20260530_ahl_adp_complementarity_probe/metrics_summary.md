# 20260530 AHL vs ADP Score Complementarity Probe

Eval-only diagnostic. Alpha and fusion threshold are selected from calibration scores only; test labels are only used for final reporting.

Robust normalization: calibration-normal median/MAD per method and stage.

## Baseline and Fusion Metrics

| Method | Stage | Alpha | Threshold | P | R | F1 | Acc | AUC-ROC | AUC-PR | TP | FP | TN | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ADP-only-DINO | S5 |  | 1.3926 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 |
| AHL-DINO | S5 |  | 0.4376 | 0.7126 | 0.8857 | 0.7898 | 0.8680 | 0.9440 | 0.9059 | 62 | 25 | 155 | 8 |
| fusion | S5 | 0.3100 | 1.9109 | 0.8714 | 0.8714 | 0.8714 | 0.9280 | 0.9789 | 0.9379 | 61 | 9 | 171 | 9 |
| ADP-only-DINO | S6 |  | 1.3926 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 |
| AHL-DINO | S6 |  | 0.5001 | 0.6875 | 0.9429 | 0.7952 | 0.8640 | 0.9560 | 0.9191 | 66 | 30 | 150 | 4 |
| fusion | S6 | 0.3300 | 2.7474 | 0.8906 | 0.8143 | 0.8507 | 0.9200 | 0.9825 | 0.9475 | 57 | 7 | 173 | 13 |
| ADP-only-DINO | S7 |  | 1.7079 | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9690 | 0.9151 | 59 | 11 | 169 | 11 |
| AHL-DINO | S7 |  | 0.5805 | 0.9592 | 0.6714 | 0.7899 | 0.9000 | 0.9515 | 0.9118 | 47 | 2 | 178 | 23 |
| fusion | S7 | 0.3600 | 2.4383 | 0.8939 | 0.8429 | 0.8676 | 0.9280 | 0.9814 | 0.9435 | 59 | 7 | 173 | 11 |
| ADP-only-DINO | S8 |  | 1.7079 | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9690 | 0.9151 | 59 | 11 | 169 | 11 |
| AHL-DINO | S8 |  | 0.4778 | 0.8485 | 0.8000 | 0.8235 | 0.9040 | 0.9362 | 0.8881 | 56 | 10 | 170 | 14 |
| fusion | S8 | 0.6000 | 3.2704 | 0.8814 | 0.7429 | 0.8062 | 0.9000 | 0.9782 | 0.9320 | 52 | 7 | 173 | 18 |

## FN Overlap

| Stage | ADP FN | AHL FN | Common FN | AHL rescues ADP FN | ADP rescues AHL FN | Fusion FN |
|---|---:|---:|---:|---:|---:|---:|
| S5 | 1 | 8 | 0 | 1 | 8 | 9 |
| S6 | 1 | 4 | 0 | 1 | 4 | 13 |
| S7 | 11 | 23 | 4 | 7 | 19 | 11 |
| S8 | 11 | 14 | 3 | 8 | 11 | 18 |

## Best Alpha

| Stage | Best Alpha | Fusion Threshold | Calib F1 | Calib Recall | Calib Precision | Test F1 | Test Recall | Test Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S5 | 0.3100 | 1.9109 | 0.8269 | 0.8776 | 0.7818 | 0.8714 | 0.8714 | 0.8714 |
| S6 | 0.3300 | 2.7474 | 0.8172 | 0.7755 | 0.8636 | 0.8507 | 0.8143 | 0.8906 |
| S7 | 0.3600 | 2.4383 | 0.7959 | 0.7959 | 0.7959 | 0.8676 | 0.8429 | 0.8939 |
| S8 | 0.6000 | 3.2704 | 0.7742 | 0.7347 | 0.8182 | 0.8062 | 0.7429 | 0.8814 |

Notes:
- `AHL rescues ADP FN` = defects missed by ADP but recovered by AHL.
- `ADP rescues AHL FN` = defects missed by AHL but recovered by ADP.
- `fusion` threshold is chosen on calibration best-F1 after robust normalization.