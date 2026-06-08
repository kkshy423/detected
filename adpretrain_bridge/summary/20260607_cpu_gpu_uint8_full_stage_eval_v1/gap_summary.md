# 20260607_cpu_gpu_uint8_full_stage_eval_v1 — Gap Summary (PASS/FAIL)

PASS = F1 gap ≤ 0.01 AND Recall gap ≤ 0.02 AND AUC-PR gap ≤ 0.005 (ref_and_query).
Final equivalence also requires pred_changed all near-threshold (see changed_near_threshold).

**Overall: 1/52 (method×stage×mode) cells PASS.**

| method | stage | mode | F1 gap | Recall gap | AUC-PR gap | pred_changed | changed_near_thr | PASS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| ADP-only-DINO | S0 | fixed_cpu_threshold | 0.0000 | 0.0000 | 0.0104 | 6 | 1 | FAIL |
| ADP-only-DINO | S0 | self_calibrated | 0.0170 | 0.0000 | 0.0104 | 6 | 1 | FAIL |
| AHL-DINO | S0 | fixed_cpu_threshold | 0.0078 | 0.0000 | 0.0017 | 7 | 1 | PASS |
| AHL-DINO | S0 | self_calibrated | 0.0163 | 0.0000 | 0.0017 | 7 | 1 | FAIL |
| ADP-only-DINO | S1 | fixed_cpu_threshold | 0.0000 | 0.0000 | 0.0104 | 6 | 1 | FAIL |
| ADP-only-DINO | S1 | self_calibrated | 0.0170 | 0.0000 | 0.0104 | 6 | 1 | FAIL |
| AHL-DINO | S1 | fixed_cpu_threshold | 0.0093 | 0.0000 | 0.0087 | 6 | 1 | FAIL |
| AHL-DINO | S1 | self_calibrated | 0.0341 | 0.0000 | 0.0087 | 6 | 1 | FAIL |
| ADP-AHL-bridge-v2 | S1 | fixed_cpu_threshold | 0.0016 | 0.0143 | 0.0095 | 5 | 0 | FAIL |
| ADP-AHL-bridge-v2 | S1 | self_calibrated | 0.0162 | 0.0000 | 0.0095 | 5 | 0 | FAIL |
| ADP-only-DINO | S2 | fixed_cpu_threshold | 0.0000 | 0.0000 | 0.0104 | 6 | 1 | FAIL |
| ADP-only-DINO | S2 | self_calibrated | 0.0170 | 0.0000 | 0.0104 | 6 | 1 | FAIL |
| AHL-DINO | S2 | fixed_cpu_threshold | 0.0017 | 0.0000 | 0.0118 | 13 | 1 | FAIL |
| AHL-DINO | S2 | self_calibrated | 0.0151 | 0.0000 | 0.0118 | 13 | 1 | FAIL |
| ADP-AHL-bridge-v2 | S2 | fixed_cpu_threshold | 0.0011 | 0.0143 | 0.0093 | 9 | 0 | FAIL |
| ADP-AHL-bridge-v2 | S2 | self_calibrated | 0.0255 | 0.0143 | 0.0093 | 9 | 0 | FAIL |
| ADP-only-DINO | S3 | fixed_cpu_threshold | 0.0046 | 0.0000 | 0.0104 | 5 | 0 | FAIL |
| ADP-only-DINO | S3 | self_calibrated | 0.0392 | 0.0286 | 0.0104 | 5 | 0 | FAIL |
| AHL-DINO | S3 | fixed_cpu_threshold | 0.0163 | 0.0000 | 0.0075 | 11 | 1 | FAIL |
| AHL-DINO | S3 | self_calibrated | 0.0064 | 0.0000 | 0.0075 | 11 | 1 | FAIL |
| ADP-AHL-bridge-v2 | S3 | fixed_cpu_threshold | 0.0072 | 0.0143 | 0.0084 | 4 | 0 | FAIL |
| ADP-AHL-bridge-v2 | S3 | self_calibrated | 0.0159 | 0.0143 | 0.0084 | 4 | 0 | FAIL |
| ADP-only-DINO | S4 | fixed_cpu_threshold | 0.0046 | 0.0000 | 0.0104 | 5 | 0 | FAIL |
| ADP-only-DINO | S4 | self_calibrated | 0.0392 | 0.0286 | 0.0104 | 5 | 0 | FAIL |
| AHL-DINO | S4 | fixed_cpu_threshold | 0.0234 | 0.0000 | 0.0102 | 17 | 1 | FAIL |
| AHL-DINO | S4 | self_calibrated | 0.0668 | 0.0000 | 0.0102 | 17 | 1 | FAIL |
| ADP-AHL-bridge-v2 | S4 | fixed_cpu_threshold | 0.0099 | 0.0000 | 0.0071 | 2 | 0 | FAIL |
| ADP-AHL-bridge-v2 | S4 | self_calibrated | 0.0283 | 0.0000 | 0.0071 | 2 | 0 | FAIL |
| ADP-only-DINO | S5 | fixed_cpu_threshold | 0.0046 | 0.0000 | 0.0104 | 5 | 0 | FAIL |
| ADP-only-DINO | S5 | self_calibrated | 0.0392 | 0.0286 | 0.0104 | 5 | 0 | FAIL |
| AHL-DINO | S5 | fixed_cpu_threshold | 0.0030 | 0.0000 | 0.0062 | 11 | 1 | FAIL |
| AHL-DINO | S5 | self_calibrated | 0.0516 | 0.0000 | 0.0062 | 11 | 1 | FAIL |
| ADP-AHL-bridge-v2 | S5 | fixed_cpu_threshold | 0.0025 | 0.0143 | 0.0056 | 3 | 0 | FAIL |
| ADP-AHL-bridge-v2 | S5 | self_calibrated | 0.0543 | 0.0143 | 0.0056 | 3 | 0 | FAIL |
| ADP-only-DINO | S6 | fixed_cpu_threshold | 0.0046 | 0.0000 | 0.0104 | 5 | 0 | FAIL |
| ADP-only-DINO | S6 | self_calibrated | 0.0392 | 0.0286 | 0.0104 | 5 | 0 | FAIL |
| AHL-DINO | S6 | fixed_cpu_threshold | 0.0184 | 0.0286 | 0.0076 | 9 | 1 | FAIL |
| AHL-DINO | S6 | self_calibrated | 0.0111 | 0.0429 | 0.0076 | 9 | 1 | FAIL |
| ADP-AHL-bridge-v2 | S6 | fixed_cpu_threshold | 0.0028 | 0.0143 | 0.0087 | 3 | 0 | FAIL |
| ADP-AHL-bridge-v2 | S6 | self_calibrated | 0.0097 | 0.0000 | 0.0087 | 3 | 0 | FAIL |
| ADP-only-DINO | S7 | fixed_cpu_threshold | 0.0000 | 0.0000 | 0.0104 | 0 | 1 | FAIL |
| ADP-only-DINO | S7 | self_calibrated | 0.0023 | 0.0143 | 0.0104 | 0 | 1 | FAIL |
| AHL-DINO | S7 | fixed_cpu_threshold | 0.0171 | 0.0286 | 0.0086 | 4 | 1 | FAIL |
| AHL-DINO | S7 | self_calibrated | 0.0259 | 0.0429 | 0.0086 | 4 | 1 | FAIL |
| ADP-AHL-bridge-v2 | S7 | fixed_cpu_threshold | 0.0326 | 0.0429 | 0.0091 | 4 | 0 | FAIL |
| ADP-AHL-bridge-v2 | S7 | self_calibrated | 0.0326 | 0.0429 | 0.0091 | 4 | 0 | FAIL |
| ADP-only-DINO | S8 | fixed_cpu_threshold | 0.0000 | 0.0000 | 0.0104 | 0 | 1 | FAIL |
| ADP-only-DINO | S8 | self_calibrated | 0.0023 | 0.0143 | 0.0104 | 0 | 1 | FAIL |
| AHL-DINO | S8 | fixed_cpu_threshold | 0.0165 | 0.0286 | 0.0119 | 6 | 1 | FAIL |
| AHL-DINO | S8 | self_calibrated | 0.0223 | 0.0286 | 0.0119 | 6 | 1 | FAIL |
| ADP-AHL-bridge-v2 | S8 | fixed_cpu_threshold | 0.0613 | 0.1000 | 0.0097 | 7 | 0 | FAIL |
| ADP-AHL-bridge-v2 | S8 | self_calibrated | 0.1093 | 0.1714 | 0.0097 | 7 | 0 | FAIL |
