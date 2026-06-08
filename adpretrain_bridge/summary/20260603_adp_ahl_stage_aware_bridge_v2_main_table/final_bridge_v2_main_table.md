# 20260603 ADP-AHL Stage-Aware Score Bridge V2 Main Table

- split: `20260529_qm_xiepai_6_1_fixed_180_70_val49`
- task type: eval/result consolidation only; no retraining; no ADP/AHL/YOLO changes
- bridge v2: `score_bridge = alpha_stage * ADP_norm + (1-alpha_stage) * AHL_norm`
- normalization: per-stage calib-normal robust median/MAD for ADP and AHL scores
- alpha: S0 not applicable; S1-S2 `0.70`; S3-S4 `0.60`; S5-S7 `0.35`; S8 `0.70`
- threshold policy: S0-S2 calibration best-F1 under `Recall>0.90`; S3-S6 under `Recall>0.85`; S7-S8 calibration best-F1
- threshold selection uses calibration labels only; test labels are only used for final reporting

| Method | Stage | Role | alpha | P | R | F1 | Acc | AUC-ROC | AUC-PR | TP | FP | TN | FN |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ADP-only-DINO | S0 | cold_start_mainline |  | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9690 | 0.9151 | 69 | 39 | 141 | 1 |
| AHL-DINO | S0 | observation_not_standalone |  | 0.4026 | 0.8857 | 0.5536 | 0.6000 | 0.8473 | 0.7653 | 62 | 92 | 88 | 8 |
| ADP-AHL bridge v2 | S0 | not_applicable_use_adp_only |  |  |  |  |  |  |  |  |  |  |  |
| YOLO26s-eval | S0 | not_trainable |  |  |  |  |  |  |  |  |  |  |  |
| ADP-only-DINO | S1 | baseline_replaced_by_bridge_v2 |  | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9690 | 0.9151 | 69 | 39 | 141 | 1 |
| AHL-DINO | S1 | observation_not_standalone |  | 0.5943 | 0.9000 | 0.7159 | 0.8000 | 0.9407 | 0.8926 | 63 | 43 | 137 | 7 |
| ADP-AHL bridge v2 | S1 | early_bridge_mainline | 0.7000 | 0.6796 | 1.0000 | 0.8092 | 0.8680 | 0.9780 | 0.9324 | 70 | 33 | 147 | 0 |
| YOLO26s-eval | S1 | supervised_observation |  | 0.2811 | 1.0000 | 0.4389 | 0.2840 | 0.8906 | 0.8139 | 70 | 179 | 1 | 0 |
| ADP-only-DINO | S2 | baseline_replaced_by_bridge_v2 |  | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9690 | 0.9151 | 69 | 39 | 141 | 1 |
| AHL-DINO | S2 | observation_not_standalone |  | 0.3681 | 0.9571 | 0.5317 | 0.5280 | 0.8650 | 0.8122 | 67 | 115 | 65 | 3 |
| ADP-AHL bridge v2 | S2 | early_bridge_mainline | 0.7000 | 0.6667 | 1.0000 | 0.8000 | 0.8600 | 0.9767 | 0.9318 | 70 | 35 | 145 | 0 |
| YOLO26s-eval | S2 | supervised_observation |  | 0.4397 | 0.8857 | 0.5877 | 0.6520 | 0.8552 | 0.7987 | 62 | 79 | 101 | 8 |
| ADP-only-DINO | S3 | baseline_replaced_by_bridge_v2 |  | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 |
| AHL-DINO | S3 | observation_not_standalone |  | 0.6598 | 0.9143 | 0.7665 | 0.8440 | 0.9308 | 0.8905 | 64 | 33 | 147 | 6 |
| ADP-AHL bridge v2 | S3 | early_bridge_mainline | 0.6000 | 0.6931 | 1.0000 | 0.8187 | 0.8760 | 0.9792 | 0.9360 | 70 | 31 | 149 | 0 |
| YOLO26s-eval | S3 | supervised_observation |  | 0.3367 | 0.9429 | 0.4962 | 0.4640 | 0.8744 | 0.8566 | 66 | 130 | 50 | 4 |
| ADP-only-DINO | S4 | baseline_replaced_by_bridge_v2 |  | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 |
| AHL-DINO | S4 | observation_not_standalone |  | 0.6436 | 0.9286 | 0.7602 | 0.8360 | 0.9392 | 0.9041 | 65 | 36 | 144 | 5 |
| ADP-AHL bridge v2 | S4 | early_bridge_mainline | 0.6000 | 0.8519 | 0.9857 | 0.9139 | 0.9480 | 0.9817 | 0.9437 | 69 | 12 | 168 | 1 |
| YOLO26s-eval | S4 | supervised_observation |  | 0.5200 | 0.9286 | 0.6667 | 0.7400 | 0.9237 | 0.9023 | 65 | 60 | 120 | 5 |
| ADP-only-DINO | S5 | fallback_baseline |  | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 |
| AHL-DINO | S5 | complementary_baseline_not_standalone |  | 0.7126 | 0.8857 | 0.7898 | 0.8680 | 0.9440 | 0.9059 | 62 | 25 | 155 | 8 |
| ADP-AHL bridge v2 | S5 | fewshot_transition_bridge_mainline | 0.3500 | 0.8767 | 0.9143 | 0.8951 | 0.9400 | 0.9803 | 0.9411 | 64 | 9 | 171 | 6 |
| YOLO26s-eval | S5 | supervised_observation |  | 0.4621 | 0.9571 | 0.6233 | 0.6760 | 0.9370 | 0.9169 | 67 | 78 | 102 | 3 |
| ADP-only-DINO | S6 | fallback_baseline |  | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 |
| AHL-DINO | S6 | complementary_baseline_not_standalone |  | 0.6875 | 0.9429 | 0.7952 | 0.8640 | 0.9560 | 0.9191 | 66 | 30 | 150 | 4 |
| ADP-AHL bridge v2 | S6 | fewshot_transition_bridge_mainline | 0.3500 | 0.8750 | 0.9000 | 0.8873 | 0.9360 | 0.9825 | 0.9471 | 63 | 9 | 171 | 7 |
| YOLO26s-eval | S6 | supervised_switch_candidate |  | 0.7727 | 0.9714 | 0.8608 | 0.9120 | 0.9817 | 0.9697 | 68 | 20 | 160 | 2 |
| ADP-only-DINO | S7 | fallback_baseline |  | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9690 | 0.9151 | 59 | 11 | 169 | 11 |
| AHL-DINO | S7 | complementary_baseline_not_standalone |  | 0.9592 | 0.6714 | 0.7899 | 0.9000 | 0.9515 | 0.9118 | 47 | 2 | 178 | 23 |
| ADP-AHL bridge v2 | S7 | fewshot_transition_bridge_mainline | 0.3500 | 0.8939 | 0.8429 | 0.8676 | 0.9280 | 0.9811 | 0.9428 | 59 | 7 | 173 | 11 |
| YOLO26s-eval | S7 | supervised_mainline |  | 0.8955 | 0.8571 | 0.8759 | 0.9320 | 0.9674 | 0.9464 | 60 | 7 | 173 | 10 |
| ADP-only-DINO | S8 | fallback_baseline |  | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9690 | 0.9151 | 59 | 11 | 169 | 11 |
| AHL-DINO | S8 | complementary_baseline_not_standalone |  | 0.8485 | 0.8000 | 0.8235 | 0.9040 | 0.9362 | 0.8881 | 56 | 10 | 170 | 14 |
| ADP-AHL bridge v2 | S8 | bridge_available_yolo_supervised_mainline | 0.7000 | 0.8923 | 0.8286 | 0.8593 | 0.9240 | 0.9761 | 0.9276 | 58 | 7 | 173 | 12 |
| YOLO26s-eval | S8 | supervised_mainline |  | 0.8986 | 0.8857 | 0.8921 | 0.9400 | 0.9790 | 0.9638 | 62 | 7 | 173 | 8 |
