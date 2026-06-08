# 20260601 Final Stage-Aware Bridge Main Table V1

- split: `20260529_qm_xiepai_6_1_fixed_180_70_val49`
- threshold source: calibration only; test labels are used only for final reporting
- threshold policy: S0-S2 calibration best-F1 under `Recall>0.90`; S3-S6 calibration best-F1 under `Recall>0.85`; S7-S8 calibration best-F1
- bridge: S5-S7 `alpha=0.35`; S8 `alpha=0.70`; S0-S4 `not_applicable`
- bridge normalization: per-stage calib-normal robust median/MAD
- source: existing eval-only outputs; no retraining; no model changes
- bridge `Total_ms`: current AHL-path latency estimate, not a newly measured end-to-end bridge latency

| Method | Stage | threshold_policy | P | R | F1 | Acc | AUC-ROC | AUC-PR | TP | FP | TN | FN | Total_ms | Main_role |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ADP-only-DINO | S0 | calib_best_f1_with_recall_gt_0.90 | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9690 | 0.9151 | 69 | 39 | 141 | 1 | 120.2927 | cold_start_mainline |
| AHL-DINO | S0 | calib_best_f1_with_recall_gt_0.90 | 0.4026 | 0.8857 | 0.5536 | 0.6000 | 0.8473 | 0.7653 | 62 | 92 | 88 | 8 | 129.5992 | few_shot_observation |
| ADP-AHL bridge v1 | S0 | calib_best_f1_with_recall_gt_0.90 |  |  |  |  |  |  |  |  |  |  |  | not_applicable |
| YOLO26s-eval | S0 | calib_best_f1_with_recall_gt_0.90 |  |  |  |  |  |  |  |  |  |  |  | not_trainable |
| ADP-only-DINO | S1 | calib_best_f1_with_recall_gt_0.90 | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9690 | 0.9151 | 69 | 39 | 141 | 1 | 118.6682 | cold_start_mainline |
| AHL-DINO | S1 | calib_best_f1_with_recall_gt_0.90 | 0.5943 | 0.9000 | 0.7159 | 0.8000 | 0.9407 | 0.8926 | 63 | 43 | 137 | 7 | 134.9796 | few_shot_observation |
| ADP-AHL bridge v1 | S1 | calib_best_f1_with_recall_gt_0.90 |  |  |  |  |  |  |  |  |  |  |  | not_applicable |
| YOLO26s-eval | S1 | calib_best_f1_with_recall_gt_0.90 | 0.2811 | 1.0000 | 0.4389 | 0.2840 | 0.8906 | 0.8139 | 70 | 179 | 1 | 0 | 22.1669 | supervised_observation |
| ADP-only-DINO | S2 | calib_best_f1_with_recall_gt_0.90 | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9690 | 0.9151 | 69 | 39 | 141 | 1 | 100.2370 | cold_start_mainline |
| AHL-DINO | S2 | calib_best_f1_with_recall_gt_0.90 | 0.3681 | 0.9571 | 0.5317 | 0.5280 | 0.8650 | 0.8122 | 67 | 115 | 65 | 3 | 132.6986 | few_shot_observation |
| ADP-AHL bridge v1 | S2 | calib_best_f1_with_recall_gt_0.90 |  |  |  |  |  |  |  |  |  |  |  | not_applicable |
| YOLO26s-eval | S2 | calib_best_f1_with_recall_gt_0.90 | 0.4397 | 0.8857 | 0.5877 | 0.6520 | 0.8552 | 0.7987 | 62 | 79 | 101 | 8 | 20.2489 | supervised_observation |
| ADP-only-DINO | S3 | calib_best_f1_with_recall_gt_0.85 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 | 98.8260 | transition_mainline |
| AHL-DINO | S3 | calib_best_f1_with_recall_gt_0.85 | 0.6598 | 0.9143 | 0.7665 | 0.8440 | 0.9308 | 0.8905 | 64 | 33 | 147 | 6 | 135.3822 | few_shot_observation |
| ADP-AHL bridge v1 | S3 | calib_best_f1_with_recall_gt_0.85 |  |  |  |  |  |  |  |  |  |  |  | not_applicable |
| YOLO26s-eval | S3 | calib_best_f1_with_recall_gt_0.85 | 0.3367 | 0.9429 | 0.4962 | 0.4640 | 0.8744 | 0.8566 | 66 | 130 | 50 | 4 | 21.6955 | supervised_observation |
| ADP-only-DINO | S4 | calib_best_f1_with_recall_gt_0.85 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 | 96.7071 | transition_mainline |
| AHL-DINO | S4 | calib_best_f1_with_recall_gt_0.85 | 0.6436 | 0.9286 | 0.7602 | 0.8360 | 0.9392 | 0.9041 | 65 | 36 | 144 | 5 | 129.4798 | few_shot_observation |
| ADP-AHL bridge v1 | S4 | calib_best_f1_with_recall_gt_0.85 |  |  |  |  |  |  |  |  |  |  |  | not_applicable |
| YOLO26s-eval | S4 | calib_best_f1_with_recall_gt_0.85 | 0.5200 | 0.9286 | 0.6667 | 0.7400 | 0.9237 | 0.9023 | 65 | 60 | 120 | 5 | 23.1483 | supervised_observation |
| ADP-only-DINO | S5 | calib_best_f1_with_recall_gt_0.85 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 | 93.5874 | baseline_and_fallback |
| AHL-DINO | S5 | calib_best_f1_with_recall_gt_0.85 | 0.7126 | 0.8857 | 0.7898 | 0.8680 | 0.9440 | 0.9059 | 62 | 25 | 155 | 8 | 134.3998 | complementary_baseline |
| ADP-AHL bridge v1 | S5 | calib_best_f1_with_recall_gt_0.85 | 0.8767 | 0.9143 | 0.8951 | 0.9400 | 0.9803 | 0.9411 | 64 | 9 | 171 | 6 | 134.3998 | stage_aware_bridge_mainline |
| YOLO26s-eval | S5 | calib_best_f1_with_recall_gt_0.85 | 0.4621 | 0.9571 | 0.6233 | 0.6760 | 0.9370 | 0.9169 | 67 | 78 | 102 | 3 | 22.4057 | supervised_observation |
| ADP-only-DINO | S6 | calib_best_f1_with_recall_gt_0.85 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 | 107.3491 | baseline_and_fallback |
| AHL-DINO | S6 | calib_best_f1_with_recall_gt_0.85 | 0.6875 | 0.9429 | 0.7952 | 0.8640 | 0.9560 | 0.9191 | 66 | 30 | 150 | 4 | 138.2637 | complementary_baseline |
| ADP-AHL bridge v1 | S6 | calib_best_f1_with_recall_gt_0.85 | 0.8750 | 0.9000 | 0.8873 | 0.9360 | 0.9825 | 0.9471 | 63 | 9 | 171 | 7 | 138.2637 | stage_aware_bridge_mainline |
| YOLO26s-eval | S6 | calib_best_f1_with_recall_gt_0.85 | 0.7727 | 0.9714 | 0.8608 | 0.9120 | 0.9817 | 0.9697 | 68 | 20 | 160 | 2 | 23.3244 | supervised_switch_candidate |
| ADP-only-DINO | S7 | calib_best_f1 | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9690 | 0.9151 | 59 | 11 | 169 | 11 | 116.0971 | baseline_and_fallback |
| AHL-DINO | S7 | calib_best_f1 | 0.9592 | 0.6714 | 0.7899 | 0.9000 | 0.9515 | 0.9118 | 47 | 2 | 178 | 23 | 132.8212 | complementary_baseline |
| ADP-AHL bridge v1 | S7 | calib_best_f1 | 0.8939 | 0.8429 | 0.8676 | 0.9280 | 0.9811 | 0.9428 | 59 | 7 | 173 | 11 | 132.8212 | stage_aware_bridge_mainline |
| YOLO26s-eval | S7 | calib_best_f1 | 0.8955 | 0.8571 | 0.8759 | 0.9320 | 0.9674 | 0.9464 | 60 | 7 | 173 | 10 | 20.3694 | supervised_mainline |
| ADP-only-DINO | S8 | calib_best_f1 | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9690 | 0.9151 | 59 | 11 | 169 | 11 | 114.7278 | baseline_and_fallback |
| AHL-DINO | S8 | calib_best_f1 | 0.8485 | 0.8000 | 0.8235 | 0.9040 | 0.9362 | 0.8881 | 56 | 10 | 170 | 14 | 138.8100 | complementary_baseline |
| ADP-AHL bridge v1 | S8 | calib_best_f1 | 0.8923 | 0.8286 | 0.8593 | 0.9240 | 0.9761 | 0.9276 | 58 | 7 | 173 | 12 | 138.8100 | stage_aware_bridge_mainline |
| YOLO26s-eval | S8 | calib_best_f1 | 0.8986 | 0.8857 | 0.8921 | 0.9400 | 0.9790 | 0.9638 | 62 | 7 | 173 | 8 | 18.7911 | supervised_mainline |

- CSV: `summary/20260601_final_stage_aware_bridge_main_table_v1/final_main_table.csv`