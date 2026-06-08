# 20260607_cpu_gpu_uint8_full_stage_eval_v1 — Master Performance Table

Rows = method × stage × backend × threshold_mode. Thresholds from calibration; test only reported.

| method | stage | backend | mode | alpha | thr | P | R | F1 | Acc | AUC-ROC | AUC-PR | TP | FP | TN | FN |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ADP-only-DINO | S0 | cpu_pil | fixed_cpu_threshold |  | 1.3744 | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9690 | 0.9151 | 69 | 39 | 141 | 1 |
| ADP-only-DINO | S0 | cpu_pil | self_calibrated |  | 1.3744 | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9690 | 0.9151 | 69 | 39 | 141 | 1 |
| ADP-only-DINO | S0 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 1.3744 | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9648 | 0.9047 | 69 | 39 | 141 | 1 |
| ADP-only-DINO | S0 | gpu_tensor_uint8_aa_true | self_calibrated |  | 1.3538 | 0.6161 | 0.9857 | 0.7582 | 0.8240 | 0.9648 | 0.9047 | 69 | 43 | 137 | 1 |
| AHL-DINO | S0 | cpu_pil | fixed_cpu_threshold |  | -0.8361 | 0.3017 | 1.0000 | 0.4636 | 0.3520 | 0.9148 | 0.8352 | 70 | 162 | 18 | 0 |
| AHL-DINO | S0 | cpu_pil | self_calibrated |  | -0.8361 | 0.3017 | 1.0000 | 0.4636 | 0.3520 | 0.9148 | 0.8352 | 70 | 162 | 18 | 0 |
| AHL-DINO | S0 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | -0.8361 | 0.3084 | 1.0000 | 0.4714 | 0.3720 | 0.9133 | 0.8335 | 70 | 157 | 23 | 0 |
| AHL-DINO | S0 | gpu_tensor_uint8_aa_true | self_calibrated |  | -0.9146 | 0.2881 | 1.0000 | 0.4473 | 0.3080 | 0.9133 | 0.8335 | 70 | 173 | 7 | 0 |
| ADP-only-DINO | S1 | cpu_pil | fixed_cpu_threshold |  | 1.3744 | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9690 | 0.9151 | 69 | 39 | 141 | 1 |
| ADP-only-DINO | S1 | cpu_pil | self_calibrated |  | 1.3744 | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9690 | 0.9151 | 69 | 39 | 141 | 1 |
| ADP-only-DINO | S1 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 1.3744 | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9648 | 0.9047 | 69 | 39 | 141 | 1 |
| ADP-only-DINO | S1 | gpu_tensor_uint8_aa_true | self_calibrated |  | 1.3538 | 0.6161 | 0.9857 | 0.7582 | 0.8240 | 0.9648 | 0.9047 | 69 | 43 | 137 | 1 |
| AHL-DINO | S1 | cpu_pil | fixed_cpu_threshold |  | 0.1361 | 0.3955 | 1.0000 | 0.5668 | 0.5720 | 0.9641 | 0.9198 | 70 | 107 | 73 | 0 |
| AHL-DINO | S1 | cpu_pil | self_calibrated |  | 0.1361 | 0.3955 | 1.0000 | 0.5668 | 0.5720 | 0.9641 | 0.9198 | 70 | 107 | 73 | 0 |
| AHL-DINO | S1 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 0.1361 | 0.4046 | 1.0000 | 0.5761 | 0.5880 | 0.9590 | 0.9110 | 70 | 103 | 77 | 0 |
| AHL-DINO | S1 | gpu_tensor_uint8_aa_true | self_calibrated |  | 0.1481 | 0.4294 | 1.0000 | 0.6009 | 0.6280 | 0.9590 | 0.9110 | 70 | 93 | 87 | 0 |
| ADP-AHL-bridge-v2 | S1 | cpu_pil | fixed_cpu_threshold | 0.70 | -0.0436 | 0.6140 | 1.0000 | 0.7609 | 0.8240 | 0.9711 | 0.9242 | 70 | 44 | 136 | 0 |
| ADP-AHL-bridge-v2 | S1 | cpu_pil | self_calibrated | 0.70 | -0.0436 | 0.6140 | 1.0000 | 0.7609 | 0.8240 | 0.9711 | 0.9242 | 70 | 44 | 136 | 0 |
| ADP-AHL-bridge-v2 | S1 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold | 0.70 | -0.0436 | 0.6216 | 0.9857 | 0.7624 | 0.8280 | 0.9682 | 0.9147 | 69 | 42 | 138 | 1 |
| ADP-AHL-bridge-v2 | S1 | gpu_tensor_uint8_aa_true | self_calibrated | 0.70 | -0.1664 | 0.5932 | 1.0000 | 0.7447 | 0.8080 | 0.9682 | 0.9147 | 70 | 48 | 132 | 0 |
| ADP-only-DINO | S2 | cpu_pil | fixed_cpu_threshold |  | 1.3744 | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9690 | 0.9151 | 69 | 39 | 141 | 1 |
| ADP-only-DINO | S2 | cpu_pil | self_calibrated |  | 1.3744 | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9690 | 0.9151 | 69 | 39 | 141 | 1 |
| ADP-only-DINO | S2 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 1.3744 | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9648 | 0.9047 | 69 | 39 | 141 | 1 |
| ADP-only-DINO | S2 | gpu_tensor_uint8_aa_true | self_calibrated |  | 1.3538 | 0.6161 | 0.9857 | 0.7582 | 0.8240 | 0.9648 | 0.9047 | 69 | 43 | 137 | 1 |
| AHL-DINO | S2 | cpu_pil | fixed_cpu_threshold |  | 0.1301 | 0.3271 | 1.0000 | 0.4930 | 0.4240 | 0.9188 | 0.8694 | 70 | 144 | 36 | 0 |
| AHL-DINO | S2 | cpu_pil | self_calibrated |  | 0.1301 | 0.3271 | 1.0000 | 0.4930 | 0.4240 | 0.9188 | 0.8694 | 70 | 144 | 36 | 0 |
| AHL-DINO | S2 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 0.1301 | 0.3256 | 1.0000 | 0.4912 | 0.4200 | 0.9123 | 0.8575 | 70 | 145 | 35 | 0 |
| AHL-DINO | S2 | gpu_tensor_uint8_aa_true | self_calibrated |  | 0.1205 | 0.3139 | 1.0000 | 0.4778 | 0.3880 | 0.9123 | 0.8575 | 70 | 153 | 27 | 0 |
| ADP-AHL-bridge-v2 | S2 | cpu_pil | fixed_cpu_threshold | 0.70 | 0.0041 | 0.5833 | 1.0000 | 0.7368 | 0.8000 | 0.9700 | 0.9214 | 70 | 50 | 130 | 0 |
| ADP-AHL-bridge-v2 | S2 | cpu_pil | self_calibrated | 0.70 | 0.0041 | 0.5833 | 1.0000 | 0.7368 | 0.8000 | 0.9700 | 0.9214 | 70 | 50 | 130 | 0 |
| ADP-AHL-bridge-v2 | S2 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold | 0.70 | 0.0041 | 0.5897 | 0.9857 | 0.7380 | 0.8040 | 0.9667 | 0.9122 | 69 | 48 | 132 | 1 |
| ADP-AHL-bridge-v2 | S2 | gpu_tensor_uint8_aa_true | self_calibrated | 0.70 | -0.0877 | 0.5565 | 0.9857 | 0.7113 | 0.7760 | 0.9667 | 0.9122 | 69 | 55 | 125 | 1 |
| ADP-only-DINO | S3 | cpu_pil | fixed_cpu_threshold |  | 1.3926 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 |
| ADP-only-DINO | S3 | cpu_pil | self_calibrated |  | 1.3926 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 |
| ADP-only-DINO | S3 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 1.3926 | 0.6699 | 0.9857 | 0.7977 | 0.8600 | 0.9648 | 0.9047 | 69 | 34 | 146 | 1 |
| ADP-only-DINO | S3 | gpu_tensor_uint8_aa_true | self_calibrated |  | 1.4570 | 0.7363 | 0.9571 | 0.8323 | 0.8920 | 0.9648 | 0.9047 | 67 | 24 | 156 | 3 |
| AHL-DINO | S3 | cpu_pil | fixed_cpu_threshold |  | 0.2406 | 0.5000 | 0.9857 | 0.6635 | 0.7200 | 0.9626 | 0.9211 | 69 | 69 | 111 | 1 |
| AHL-DINO | S3 | cpu_pil | self_calibrated |  | 0.2406 | 0.5000 | 0.9857 | 0.6635 | 0.7200 | 0.9626 | 0.9211 | 69 | 69 | 111 | 1 |
| AHL-DINO | S3 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 0.2406 | 0.5188 | 0.9857 | 0.6798 | 0.7400 | 0.9599 | 0.9137 | 69 | 64 | 116 | 1 |
| AHL-DINO | S3 | gpu_tensor_uint8_aa_true | self_calibrated |  | 0.2373 | 0.5074 | 0.9857 | 0.6699 | 0.7280 | 0.9599 | 0.9137 | 69 | 67 | 113 | 1 |
| ADP-AHL-bridge-v2 | S3 | cpu_pil | fixed_cpu_threshold | 0.60 | 0.5340 | 0.6800 | 0.9714 | 0.8000 | 0.8640 | 0.9719 | 0.9277 | 68 | 32 | 148 | 2 |
| ADP-AHL-bridge-v2 | S3 | cpu_pil | self_calibrated | 0.60 | 0.5340 | 0.6800 | 0.9714 | 0.8000 | 0.8640 | 0.9719 | 0.9277 | 68 | 32 | 148 | 2 |
| ADP-AHL-bridge-v2 | S3 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold | 0.60 | 0.5340 | 0.6979 | 0.9571 | 0.8072 | 0.8720 | 0.9693 | 0.9193 | 67 | 29 | 151 | 3 |
| ADP-AHL-bridge-v2 | S3 | gpu_tensor_uint8_aa_true | self_calibrated | 0.60 | 0.1666 | 0.6509 | 0.9857 | 0.7841 | 0.8480 | 0.9693 | 0.9193 | 69 | 37 | 143 | 1 |
| ADP-only-DINO | S4 | cpu_pil | fixed_cpu_threshold |  | 1.3926 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 |
| ADP-only-DINO | S4 | cpu_pil | self_calibrated |  | 1.3926 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 |
| ADP-only-DINO | S4 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 1.3926 | 0.6699 | 0.9857 | 0.7977 | 0.8600 | 0.9648 | 0.9047 | 69 | 34 | 146 | 1 |
| ADP-only-DINO | S4 | gpu_tensor_uint8_aa_true | self_calibrated |  | 1.4570 | 0.7363 | 0.9571 | 0.8323 | 0.8920 | 0.9648 | 0.9047 | 67 | 24 | 156 | 3 |
| AHL-DINO | S4 | cpu_pil | fixed_cpu_threshold |  | 0.4102 | 0.4182 | 0.9857 | 0.5872 | 0.6120 | 0.9586 | 0.9220 | 69 | 96 | 84 | 1 |
| AHL-DINO | S4 | cpu_pil | self_calibrated |  | 0.4102 | 0.4182 | 0.9857 | 0.5872 | 0.6120 | 0.9586 | 0.9220 | 69 | 96 | 84 | 1 |
| AHL-DINO | S4 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 0.4102 | 0.4423 | 0.9857 | 0.6106 | 0.6480 | 0.9540 | 0.9117 | 69 | 87 | 93 | 1 |
| AHL-DINO | S4 | gpu_tensor_uint8_aa_true | self_calibrated |  | 0.4178 | 0.4894 | 0.9857 | 0.6540 | 0.7080 | 0.9540 | 0.9117 | 69 | 72 | 108 | 1 |
| ADP-AHL-bridge-v2 | S4 | cpu_pil | fixed_cpu_threshold | 0.60 | 0.4568 | 0.7041 | 0.9857 | 0.8214 | 0.8800 | 0.9728 | 0.9285 | 69 | 29 | 151 | 1 |
| ADP-AHL-bridge-v2 | S4 | cpu_pil | self_calibrated | 0.60 | 0.4568 | 0.7041 | 0.9857 | 0.8214 | 0.8800 | 0.9728 | 0.9285 | 69 | 29 | 151 | 1 |
| ADP-AHL-bridge-v2 | S4 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold | 0.60 | 0.4568 | 0.7188 | 0.9857 | 0.8313 | 0.8880 | 0.9703 | 0.9214 | 69 | 27 | 153 | 1 |
| ADP-AHL-bridge-v2 | S4 | gpu_tensor_uint8_aa_true | self_calibrated | 0.60 | 0.0424 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9703 | 0.9214 | 69 | 35 | 145 | 1 |
| ADP-only-DINO | S5 | cpu_pil | fixed_cpu_threshold |  | 1.3926 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 |
| ADP-only-DINO | S5 | cpu_pil | self_calibrated |  | 1.3926 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 |
| ADP-only-DINO | S5 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 1.3926 | 0.6699 | 0.9857 | 0.7977 | 0.8600 | 0.9648 | 0.9047 | 69 | 34 | 146 | 1 |
| ADP-only-DINO | S5 | gpu_tensor_uint8_aa_true | self_calibrated |  | 1.4570 | 0.7363 | 0.9571 | 0.8323 | 0.8920 | 0.9648 | 0.9047 | 67 | 24 | 156 | 3 |
| AHL-DINO | S5 | cpu_pil | fixed_cpu_threshold |  | 0.3886 | 0.4759 | 0.9857 | 0.6419 | 0.6920 | 0.9660 | 0.9255 | 69 | 76 | 104 | 1 |
| AHL-DINO | S5 | cpu_pil | self_calibrated |  | 0.3886 | 0.4759 | 0.9857 | 0.6419 | 0.6920 | 0.9660 | 0.9255 | 69 | 76 | 104 | 1 |
| AHL-DINO | S5 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 0.3886 | 0.4792 | 0.9857 | 0.6449 | 0.6960 | 0.9625 | 0.9193 | 69 | 75 | 105 | 1 |
| AHL-DINO | S5 | gpu_tensor_uint8_aa_true | self_calibrated |  | 0.4009 | 0.5349 | 0.9857 | 0.6935 | 0.7560 | 0.9625 | 0.9193 | 69 | 60 | 120 | 1 |
| ADP-AHL-bridge-v2 | S5 | cpu_pil | fixed_cpu_threshold | 0.35 | 0.2072 | 0.6832 | 0.9857 | 0.8070 | 0.8680 | 0.9731 | 0.9327 | 69 | 32 | 148 | 1 |
| ADP-AHL-bridge-v2 | S5 | cpu_pil | self_calibrated | 0.35 | 0.2072 | 0.6832 | 0.9857 | 0.8070 | 0.8680 | 0.9731 | 0.9327 | 69 | 32 | 148 | 1 |
| ADP-AHL-bridge-v2 | S5 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold | 0.35 | 0.2072 | 0.6939 | 0.9714 | 0.8095 | 0.8720 | 0.9711 | 0.9271 | 68 | 30 | 150 | 2 |
| ADP-AHL-bridge-v2 | S5 | gpu_tensor_uint8_aa_true | self_calibrated | 0.35 | -0.1928 | 0.6034 | 1.0000 | 0.7527 | 0.8160 | 0.9711 | 0.9271 | 70 | 46 | 134 | 0 |
| ADP-only-DINO | S6 | cpu_pil | fixed_cpu_threshold |  | 1.3926 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 |
| ADP-only-DINO | S6 | cpu_pil | self_calibrated |  | 1.3926 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 |
| ADP-only-DINO | S6 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 1.3926 | 0.6699 | 0.9857 | 0.7977 | 0.8600 | 0.9648 | 0.9047 | 69 | 34 | 146 | 1 |
| ADP-only-DINO | S6 | gpu_tensor_uint8_aa_true | self_calibrated |  | 1.4570 | 0.7363 | 0.9571 | 0.8323 | 0.8920 | 0.9648 | 0.9047 | 67 | 24 | 156 | 3 |
| AHL-DINO | S6 | cpu_pil | fixed_cpu_threshold |  | 0.5132 | 0.6509 | 0.9857 | 0.7841 | 0.8480 | 0.9629 | 0.9263 | 69 | 37 | 143 | 1 |
| AHL-DINO | S6 | cpu_pil | self_calibrated |  | 0.5132 | 0.6509 | 0.9857 | 0.7841 | 0.8480 | 0.9629 | 0.9263 | 69 | 37 | 143 | 1 |
| AHL-DINO | S6 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 0.5132 | 0.6381 | 0.9571 | 0.7657 | 0.8360 | 0.9584 | 0.9187 | 67 | 38 | 142 | 3 |
| AHL-DINO | S6 | gpu_tensor_uint8_aa_true | self_calibrated |  | 0.5248 | 0.6875 | 0.9429 | 0.7952 | 0.8640 | 0.9584 | 0.9187 | 66 | 30 | 150 | 4 |
| ADP-AHL-bridge-v2 | S6 | cpu_pil | fixed_cpu_threshold | 0.35 | 0.3054 | 0.7041 | 0.9857 | 0.8214 | 0.8800 | 0.9752 | 0.9359 | 69 | 29 | 151 | 1 |
| ADP-AHL-bridge-v2 | S6 | cpu_pil | self_calibrated | 0.35 | 0.3054 | 0.7041 | 0.9857 | 0.8214 | 0.8800 | 0.9752 | 0.9359 | 69 | 29 | 151 | 1 |
| ADP-AHL-bridge-v2 | S6 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold | 0.35 | 0.3054 | 0.7158 | 0.9714 | 0.8242 | 0.8840 | 0.9713 | 0.9271 | 68 | 27 | 153 | 2 |
| ADP-AHL-bridge-v2 | S6 | gpu_tensor_uint8_aa_true | self_calibrated | 0.35 | 0.1356 | 0.6900 | 0.9857 | 0.8118 | 0.8720 | 0.9713 | 0.9271 | 69 | 31 | 149 | 1 |
| ADP-only-DINO | S7 | cpu_pil | fixed_cpu_threshold |  | 1.7079 | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9690 | 0.9151 | 59 | 11 | 169 | 11 |
| ADP-only-DINO | S7 | cpu_pil | self_calibrated |  | 1.7079 | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9690 | 0.9151 | 59 | 11 | 169 | 11 |
| ADP-only-DINO | S7 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 1.7079 | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9648 | 0.9047 | 59 | 11 | 169 | 11 |
| ADP-only-DINO | S7 | gpu_tensor_uint8_aa_true | self_calibrated |  | 1.7468 | 0.8529 | 0.8286 | 0.8406 | 0.9120 | 0.9648 | 0.9047 | 58 | 10 | 170 | 12 |
| AHL-DINO | S7 | cpu_pil | fixed_cpu_threshold |  | 0.5351 | 0.8788 | 0.8286 | 0.8529 | 0.9200 | 0.9535 | 0.9103 | 58 | 8 | 172 | 12 |
| AHL-DINO | S7 | cpu_pil | self_calibrated |  | 0.5351 | 0.8788 | 0.8286 | 0.8529 | 0.9200 | 0.9535 | 0.9103 | 58 | 8 | 172 | 12 |
| AHL-DINO | S7 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 0.5351 | 0.8750 | 0.8000 | 0.8358 | 0.9120 | 0.9498 | 0.9017 | 56 | 8 | 172 | 14 |
| AHL-DINO | S7 | gpu_tensor_uint8_aa_true | self_calibrated |  | 0.5397 | 0.8730 | 0.7857 | 0.8271 | 0.9080 | 0.9498 | 0.9017 | 55 | 8 | 172 | 15 |
| ADP-AHL-bridge-v2 | S7 | cpu_pil | fixed_cpu_threshold | 0.35 | 2.4871 | 0.8906 | 0.8143 | 0.8507 | 0.9200 | 0.9721 | 0.9270 | 57 | 7 | 173 | 13 |
| ADP-AHL-bridge-v2 | S7 | cpu_pil | self_calibrated | 0.35 | 2.4871 | 0.8906 | 0.8143 | 0.8507 | 0.9200 | 0.9721 | 0.9270 | 57 | 7 | 173 | 13 |
| ADP-AHL-bridge-v2 | S7 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold | 0.35 | 2.4871 | 0.8710 | 0.7714 | 0.8182 | 0.9040 | 0.9678 | 0.9179 | 54 | 8 | 172 | 16 |
| ADP-AHL-bridge-v2 | S7 | gpu_tensor_uint8_aa_true | self_calibrated | 0.35 | 2.3742 | 0.8710 | 0.7714 | 0.8182 | 0.9040 | 0.9678 | 0.9179 | 54 | 8 | 172 | 16 |
| ADP-only-DINO | S8 | cpu_pil | fixed_cpu_threshold |  | 1.7079 | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9690 | 0.9151 | 59 | 11 | 169 | 11 |
| ADP-only-DINO | S8 | cpu_pil | self_calibrated |  | 1.7079 | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9690 | 0.9151 | 59 | 11 | 169 | 11 |
| ADP-only-DINO | S8 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 1.7079 | 0.8429 | 0.8429 | 0.8429 | 0.9120 | 0.9648 | 0.9047 | 59 | 11 | 169 | 11 |
| ADP-only-DINO | S8 | gpu_tensor_uint8_aa_true | self_calibrated |  | 1.7468 | 0.8529 | 0.8286 | 0.8406 | 0.9120 | 0.9648 | 0.9047 | 58 | 10 | 170 | 12 |
| AHL-DINO | S8 | cpu_pil | fixed_cpu_threshold |  | 0.4915 | 0.8219 | 0.8571 | 0.8392 | 0.9080 | 0.9629 | 0.9150 | 60 | 13 | 167 | 10 |
| AHL-DINO | S8 | cpu_pil | self_calibrated |  | 0.4915 | 0.8219 | 0.8571 | 0.8392 | 0.9080 | 0.9629 | 0.9150 | 60 | 13 | 167 | 10 |
| AHL-DINO | S8 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 0.4915 | 0.8169 | 0.8286 | 0.8227 | 0.9000 | 0.9577 | 0.9031 | 58 | 13 | 167 | 12 |
| AHL-DINO | S8 | gpu_tensor_uint8_aa_true | self_calibrated |  | 0.4886 | 0.8056 | 0.8286 | 0.8169 | 0.8960 | 0.9577 | 0.9031 | 58 | 14 | 166 | 12 |
| ADP-AHL-bridge-v2 | S8 | cpu_pil | fixed_cpu_threshold | 0.70 | 2.6908 | 0.8806 | 0.8429 | 0.8613 | 0.9240 | 0.9729 | 0.9275 | 59 | 8 | 172 | 11 |
| ADP-AHL-bridge-v2 | S8 | cpu_pil | self_calibrated | 0.70 | 2.6908 | 0.8806 | 0.8429 | 0.8613 | 0.9240 | 0.9729 | 0.9275 | 59 | 8 | 172 | 11 |
| ADP-AHL-bridge-v2 | S8 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold | 0.70 | 2.6908 | 0.8667 | 0.7429 | 0.8000 | 0.8960 | 0.9692 | 0.9178 | 52 | 8 | 172 | 18 |
| ADP-AHL-bridge-v2 | S8 | gpu_tensor_uint8_aa_true | self_calibrated | 0.70 | 3.0940 | 0.8545 | 0.6714 | 0.7520 | 0.8760 | 0.9692 | 0.9178 | 47 | 8 | 172 | 23 |
