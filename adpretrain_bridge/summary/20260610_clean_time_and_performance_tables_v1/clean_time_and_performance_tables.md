# Clean Time and Performance Tables 20260610

## Scope

- Dataset/split: models__qiumianxiepai, fixed 180N/70D test, calib-selected thresholds.
- Timing excludes image loading/decode and file saving. Unit is single-image ms.
- ADP/AHL timing uses median core phases from existing paired uint8 preprocessing benchmarks; all reported core totals are under 50 ms.
- YOLO metrics use the eval-only primary policy in each stage metrics.json. S0 is not trainable because no anomaly sample exists in training.

## Clean End-to-End Time Table

| method | backend | core_total_ms_median | preprocess_ms | encoder_ms | adp_match_ms | residual_ms | projector_ms | compress_ms | ahl_forward_ms | postprocess_threshold_ms | source_check_total_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADP-only-DINO | cpu_pil | 38.74 | 19.38 | 18.36 | 0.80 | 0.08 |  |  |  | 0.12 | 48.43 |
| ADP-only-DINO | gpu_tensor_uint8_aa_true | 24.10 | 4.92 | 18.23 | 0.76 | 0.07 |  |  |  | 0.12 | 33.64 |
| AHL-DINO | cpu_pil | 47.93 | 19.38 | 18.36 |  |  | 7.45 | 0.84 | 1.72 | 0.12 | 48.43 |
| AHL-DINO | gpu_tensor_uint8_aa_true | 33.26 | 4.92 | 18.23 |  |  | 7.45 | 0.84 | 1.64 | 0.12 | 33.64 |

## Preprocess Detail

| backend | phase | median_ms |
| --- | --- | --- |
| cpu_pil | cpu_resize_ms | 17.98 |
| cpu_pil | cpu_crop_ms | 0.11 |
| cpu_pil | cpu_to_tensor_ms | 0.75 |
| cpu_pil | cpu_normalize_ms | 0.31 |
| cpu_pil | h2d_copy_ms | 0.22 |
| gpu_tensor_uint8_aa_true | pil_to_tensor_cpu_ms | 3.46 |
| gpu_tensor_uint8_aa_true | raw_h2d_copy_ms | 1.06 |
| gpu_tensor_uint8_aa_true | gpu_cast_scale_ms | 0.18 |
| gpu_tensor_uint8_aa_true | gpu_resize_ms | 0.37 |
| gpu_tensor_uint8_aa_true | gpu_center_crop_ms | 0.06 |
| gpu_tensor_uint8_aa_true | gpu_normalize_ms | 0.06 |

## Clean Stage Performance Table

| stage | method | backend | precision | recall | accuracy | f1 | auc_pr | time_ms | tp | fp | tn | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | ADP-only | cpu_pil | 0.6389 | 0.9857 | 0.8400 | 0.7753 | 0.9151 | 38.74 | 69 | 39 | 141 | 1 |
| S0 | ADP-only | gpu_uint8 | 0.6161 | 0.9857 | 0.8240 | 0.7582 | 0.9047 | 24.10 | 69 | 43 | 137 | 1 |
| S0 | AHL | cpu_pil | 0.4026 | 0.8857 | 0.6000 | 0.5536 | 0.7653 | 47.93 | 62 | 92 | 88 | 8 |
| S0 | AHL | gpu_uint8 | 0.2805 | 0.9857 | 0.2880 | 0.4367 | 0.2133 | 33.26 | 69 | 177 | 3 | 1 |
| S0 | YOLO26s | gpu_eval_only |  |  |  |  |  |  |  |  |  |  |
| S1 | ADP-only | cpu_pil | 0.6389 | 0.9857 | 0.8400 | 0.7753 | 0.9151 | 38.74 | 69 | 39 | 141 | 1 |
| S1 | ADP-only | gpu_uint8 | 0.6161 | 0.9857 | 0.8240 | 0.7582 | 0.9047 | 24.10 | 69 | 43 | 137 | 1 |
| S1 | AHL | cpu_pil | 0.5943 | 0.9000 | 0.8000 | 0.7159 | 0.8926 | 47.93 | 63 | 43 | 137 | 7 |
| S1 | AHL | gpu_uint8 | 0.6238 | 0.9000 | 0.8200 | 0.7368 | 0.8968 | 33.26 | 63 | 38 | 142 | 7 |
| S1 | YOLO26s | gpu_eval_only | 0.2811 | 1.0000 | 0.2840 | 0.4389 | 0.8139 | 22.17 | 70 | 179 | 1 | 0 |
| S2 | ADP-only | cpu_pil | 0.6389 | 0.9857 | 0.8400 | 0.7753 | 0.9151 | 38.74 | 69 | 39 | 141 | 1 |
| S2 | ADP-only | gpu_uint8 | 0.6161 | 0.9857 | 0.8240 | 0.7582 | 0.9047 | 24.10 | 69 | 43 | 137 | 1 |
| S2 | AHL | cpu_pil | 0.3681 | 0.9571 | 0.5280 | 0.5317 | 0.8122 | 47.93 | 67 | 115 | 65 | 3 |
| S2 | AHL | gpu_uint8 | 0.4156 | 0.9143 | 0.6160 | 0.5714 | 0.8326 | 33.26 | 64 | 90 | 90 | 6 |
| S2 | YOLO26s | gpu_eval_only | 0.4397 | 0.8857 | 0.6520 | 0.5877 | 0.7987 | 20.25 | 62 | 79 | 101 | 8 |
| S3 | ADP-only | cpu_pil | 0.6635 | 0.9857 | 0.8560 | 0.7931 | 0.9151 | 38.74 | 69 | 35 | 145 | 1 |
| S3 | ADP-only | gpu_uint8 | 0.7363 | 0.9571 | 0.8920 | 0.8323 | 0.9047 | 24.10 | 67 | 24 | 156 | 3 |
| S3 | AHL | cpu_pil | 0.6598 | 0.9143 | 0.8440 | 0.7665 | 0.8905 | 47.93 | 64 | 33 | 147 | 6 |
| S3 | AHL | gpu_uint8 | 0.6737 | 0.9143 | 0.8520 | 0.7758 | 0.9041 | 33.26 | 64 | 31 | 149 | 6 |
| S3 | YOLO26s | gpu_eval_only | 0.3367 | 0.9429 | 0.4640 | 0.4962 | 0.8566 | 21.70 | 66 | 130 | 50 | 4 |
| S4 | ADP-only | cpu_pil | 0.6635 | 0.9857 | 0.8560 | 0.7931 | 0.9151 | 38.74 | 69 | 35 | 145 | 1 |
| S4 | ADP-only | gpu_uint8 | 0.7363 | 0.9571 | 0.8920 | 0.8323 | 0.9047 | 24.10 | 67 | 24 | 156 | 3 |
| S4 | AHL | cpu_pil | 0.6436 | 0.9286 | 0.8360 | 0.7602 | 0.9041 | 47.93 | 65 | 36 | 144 | 5 |
| S4 | AHL | gpu_uint8 | 0.6702 | 0.9000 | 0.8480 | 0.7683 | 0.9028 | 33.26 | 63 | 31 | 149 | 7 |
| S4 | YOLO26s | gpu_eval_only | 0.5200 | 0.9286 | 0.7400 | 0.6667 | 0.9023 | 23.15 | 65 | 60 | 120 | 5 |
| S5 | ADP-only | cpu_pil | 0.6635 | 0.9857 | 0.8560 | 0.7931 | 0.9151 | 38.74 | 69 | 35 | 145 | 1 |
| S5 | ADP-only | gpu_uint8 | 0.7363 | 0.9571 | 0.8920 | 0.8323 | 0.9047 | 24.10 | 67 | 24 | 156 | 3 |
| S5 | AHL | cpu_pil | 0.7126 | 0.8857 | 0.8680 | 0.7898 | 0.9059 | 47.93 | 62 | 25 | 155 | 8 |
| S5 | AHL | gpu_uint8 | 0.4779 | 0.9286 | 0.6960 | 0.6311 | 0.8796 | 33.26 | 65 | 71 | 109 | 5 |
| S5 | YOLO26s | gpu_eval_only | 0.4621 | 0.9571 | 0.6760 | 0.6233 | 0.9169 | 22.41 | 67 | 78 | 102 | 3 |
| S6 | ADP-only | cpu_pil | 0.6635 | 0.9857 | 0.8560 | 0.7931 | 0.9151 | 38.74 | 69 | 35 | 145 | 1 |
| S6 | ADP-only | gpu_uint8 | 0.7363 | 0.9571 | 0.8920 | 0.8323 | 0.9047 | 24.10 | 67 | 24 | 156 | 3 |
| S6 | AHL | cpu_pil | 0.6875 | 0.9429 | 0.8640 | 0.7952 | 0.9191 | 47.93 | 66 | 30 | 150 | 4 |
| S6 | AHL | gpu_uint8 | 0.5702 | 0.9286 | 0.7840 | 0.7065 | 0.8898 | 33.26 | 65 | 49 | 131 | 5 |
| S6 | YOLO26s | gpu_eval_only | 0.7727 | 0.9714 | 0.9120 | 0.8608 | 0.9697 | 23.32 | 68 | 20 | 160 | 2 |
| S7 | ADP-only | cpu_pil | 0.8429 | 0.8429 | 0.9120 | 0.8429 | 0.9151 | 38.74 | 59 | 11 | 169 | 11 |
| S7 | ADP-only | gpu_uint8 | 0.8529 | 0.8286 | 0.9120 | 0.8406 | 0.9047 | 24.10 | 58 | 10 | 170 | 12 |
| S7 | AHL | cpu_pil | 0.9592 | 0.6714 | 0.9000 | 0.7899 | 0.9118 | 47.93 | 47 | 2 | 178 | 23 |
| S7 | AHL | gpu_uint8 | 0.8750 | 0.8000 | 0.9120 | 0.8358 | 0.9000 | 33.26 | 56 | 8 | 172 | 14 |
| S7 | YOLO26s | gpu_eval_only | 0.8955 | 0.8571 | 0.9320 | 0.8759 | 0.9464 | 20.37 | 60 | 7 | 173 | 10 |
| S8 | ADP-only | cpu_pil | 0.8429 | 0.8429 | 0.9120 | 0.8429 | 0.9151 | 38.74 | 59 | 11 | 169 | 11 |
| S8 | ADP-only | gpu_uint8 | 0.8529 | 0.8286 | 0.9120 | 0.8406 | 0.9047 | 24.10 | 58 | 10 | 170 | 12 |
| S8 | AHL | cpu_pil | 0.8485 | 0.8000 | 0.9040 | 0.8235 | 0.8881 | 47.93 | 56 | 10 | 170 | 14 |
| S8 | AHL | gpu_uint8 | 0.9500 | 0.5429 | 0.8640 | 0.6909 | 0.8887 | 33.26 | 38 | 2 | 178 | 32 |
| S8 | YOLO26s | gpu_eval_only | 0.8986 | 0.8857 | 0.9400 | 0.8921 | 0.9638 | 18.79 | 62 | 7 | 173 | 8 |

## Source Files

- summary/20260607_gpu_preprocess_uint8_equiv_v1/equivalence_fix_latency.csv
- summary/20260606_preprocess_equivalence_and_paired_benchmark_v1_latency.csv
- summary/20260609_cpu_gpu_full_flow_rerun_v1/cpu_gpu_full_flow_master_table.csv
- output/20260529_yolo26s_cls_fixed_180_70_val49_eval_only/stages/S*/metrics/metrics.json
