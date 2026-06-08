# 20260530 Stage-Aware ADP/AHL Bridge V1

- split: `20260529_qm_xiepai_6_1_fixed_180_70_val49`
- normalization: ADP/AHL scores are robust-normalized by calibration-normal median/MAD per stage
- fixed bridge: `score = alpha * ADP_norm + (1-alpha) * AHL_norm`
- alpha: S5-S7 use `0.35`; S8 reports `0.70` and ADP-only fallback
- threshold policy: S5-S6 use calibration best-F1 under `recall>=0.85`; S7-S8 use calibration best-F1
- source: existing ADP-only-DINO and AHL-DINO score files; no retraining; no ADP/AHL model changes

| Method | Stage | Status | P | R | F1 | Acc | AUC-ROC | AUC-PR | TP | FP | TN | FN | Total ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bridge_alpha_0.35 | S5 | ok | 0.8767 | 0.9143 | 0.8951 | 0.9400 | 0.9803 | 0.9411 | 64 | 9 | 171 | 6 | 134.3998 |
| ADP-only-DINO | S5 | ok | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 | 93.5874 |
| AHL-DINO | S5 | ok | 0.7126 | 0.8857 | 0.7898 | 0.8680 | 0.9440 | 0.9059 | 62 | 25 | 155 | 8 | 134.3998 |
| bridge_alpha_0.35 | S6 | ok | 0.8750 | 0.9000 | 0.8873 | 0.9360 | 0.9825 | 0.9471 | 63 | 9 | 171 | 7 | 138.2637 |
| ADP-only-DINO | S6 | ok | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 | 107.3491 |
| AHL-DINO | S6 | ok | 0.6875 | 0.9429 | 0.7952 | 0.8640 | 0.9560 | 0.9191 | 66 | 30 | 150 | 4 | 138.2637 |
| bridge_alpha_0.35 | S7 | ok | 0.8939 | 0.8429 | 0.8676 | 0.9280 | 0.9811 | 0.9428 | 59 | 7 | 173 | 11 | 132.8212 |
| ADP-only-DINO | S7 | ok | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9690 | 0.9151 | 59 | 11 | 169 | 11 | 116.0971 |
| AHL-DINO | S7 | ok | 0.9592 | 0.6714 | 0.7899 | 0.9000 | 0.9515 | 0.9118 | 47 | 2 | 178 | 23 | 132.8212 |
| bridge_alpha_0.70 | S8 | ok | 0.8923 | 0.8286 | 0.8593 | 0.9240 | 0.9761 | 0.9276 | 58 | 7 | 173 | 12 | 138.8100 |
| ADP-only-DINO | S8 | ok | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9690 | 0.9151 | 59 | 11 | 169 | 11 | 114.7278 |
| AHL-DINO | S8 | ok | 0.8485 | 0.8000 | 0.8235 | 0.9040 | 0.9362 | 0.8881 | 56 | 10 | 170 | 14 | 138.8100 |
| ADP-only fallback | S8 | ok | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9690 | 0.9151 | 59 | 11 | 169 | 11 | 114.7278 |

- CSV: `summary/20260530_stage_aware_adp_ahl_bridge_v1/metrics_summary.csv`