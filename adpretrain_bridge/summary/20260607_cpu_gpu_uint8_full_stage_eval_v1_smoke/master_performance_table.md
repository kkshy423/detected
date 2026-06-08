# 20260607_cpu_gpu_uint8_full_stage_eval_v1 — Master Performance Table

Rows = method × stage × backend × threshold_mode. Thresholds from calibration; test only reported.

| method | stage | backend | mode | alpha | thr | P | R | F1 | Acc | AUC-ROC | AUC-PR | TP | FP | TN | FN |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ADP-only-DINO | S7 | cpu_pil | fixed_cpu_threshold |  | 1.5724 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 0.9975 | 0.9976 | 18 | 0 | 20 | 2 |
| ADP-only-DINO | S7 | cpu_pil | self_calibrated |  | 1.5724 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 0.9975 | 0.9976 | 18 | 0 | 20 | 2 |
| ADP-only-DINO | S7 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 1.5724 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 0.9950 | 0.9955 | 18 | 0 | 20 | 2 |
| ADP-only-DINO | S7 | gpu_tensor_uint8_aa_true | self_calibrated |  | 1.5717 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 0.9950 | 0.9955 | 18 | 0 | 20 | 2 |
| AHL-DINO | S7 | cpu_pil | fixed_cpu_threshold |  | 0.4879 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9800 | 0.9817 | 18 | 2 | 18 | 2 |
| AHL-DINO | S7 | cpu_pil | self_calibrated |  | 0.4879 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9800 | 0.9817 | 18 | 2 | 18 | 2 |
| AHL-DINO | S7 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 0.4879 | 0.8947 | 0.8500 | 0.8718 | 0.8750 | 0.9775 | 0.9798 | 17 | 2 | 18 | 3 |
| AHL-DINO | S7 | gpu_tensor_uint8_aa_true | self_calibrated |  | 0.4585 | 0.8571 | 0.9000 | 0.8780 | 0.8750 | 0.9775 | 0.9798 | 18 | 3 | 17 | 2 |
| ADP-AHL-bridge-v2 | S7 | cpu_pil | fixed_cpu_threshold | 0.35 | 6.1055 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 0.9900 | 0.9907 | 18 | 0 | 20 | 2 |
| ADP-AHL-bridge-v2 | S7 | cpu_pil | self_calibrated | 0.35 | 6.1055 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 0.9900 | 0.9907 | 18 | 0 | 20 | 2 |
| ADP-AHL-bridge-v2 | S7 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold | 0.35 | 6.1055 | 1.0000 | 0.8500 | 0.9189 | 0.9250 | 0.9900 | 0.9907 | 17 | 0 | 20 | 3 |
| ADP-AHL-bridge-v2 | S7 | gpu_tensor_uint8_aa_true | self_calibrated | 0.35 | 5.2976 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 0.9900 | 0.9907 | 18 | 0 | 20 | 2 |
| ADP-only-DINO | S8 | cpu_pil | fixed_cpu_threshold |  | 1.5724 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 0.9975 | 0.9976 | 18 | 0 | 20 | 2 |
| ADP-only-DINO | S8 | cpu_pil | self_calibrated |  | 1.5724 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 0.9975 | 0.9976 | 18 | 0 | 20 | 2 |
| ADP-only-DINO | S8 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 1.5724 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 0.9950 | 0.9955 | 18 | 0 | 20 | 2 |
| ADP-only-DINO | S8 | gpu_tensor_uint8_aa_true | self_calibrated |  | 1.5717 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 0.9950 | 0.9955 | 18 | 0 | 20 | 2 |
| AHL-DINO | S8 | cpu_pil | fixed_cpu_threshold |  | 0.4302 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 20 | 0 | 20 | 0 |
| AHL-DINO | S8 | cpu_pil | self_calibrated |  | 0.4302 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 20 | 0 | 20 | 0 |
| AHL-DINO | S8 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold |  | 0.4302 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 20 | 0 | 20 | 0 |
| AHL-DINO | S8 | gpu_tensor_uint8_aa_true | self_calibrated |  | 0.4369 | 1.0000 | 0.9500 | 0.9744 | 0.9750 | 1.0000 | 1.0000 | 19 | 0 | 20 | 1 |
| ADP-AHL-bridge-v2 | S8 | cpu_pil | fixed_cpu_threshold | 0.70 | 4.6003 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 1.0000 | 1.0000 | 18 | 0 | 20 | 2 |
| ADP-AHL-bridge-v2 | S8 | cpu_pil | self_calibrated | 0.70 | 4.6003 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 1.0000 | 1.0000 | 18 | 0 | 20 | 2 |
| ADP-AHL-bridge-v2 | S8 | gpu_tensor_uint8_aa_true | fixed_cpu_threshold | 0.70 | 4.6003 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 0.9975 | 0.9976 | 18 | 0 | 20 | 2 |
| ADP-AHL-bridge-v2 | S8 | gpu_tensor_uint8_aa_true | self_calibrated | 0.70 | 4.9652 | 1.0000 | 0.9000 | 0.9474 | 0.9500 | 0.9975 | 0.9976 | 18 | 0 | 20 | 2 |
