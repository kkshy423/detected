# 20260607_cpu_gpu_uint8_full_stage_eval_v1 — Gap Summary (PASS/FAIL)

PASS = F1 gap ≤ 0.01 AND Recall gap ≤ 0.02 AND AUC-PR gap ≤ 0.005 (ref_and_query).
Final equivalence also requires pred_changed all near-threshold (see changed_near_threshold).

**Overall: 8/12 (method×stage×mode) cells PASS.**

| method | stage | mode | F1 gap | Recall gap | AUC-PR gap | pred_changed | changed_near_thr | PASS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| ADP-only-DINO | S7 | fixed_cpu_threshold | 0.0000 | 0.0000 | 0.0022 | 0 | 1 | PASS |
| ADP-only-DINO | S7 | self_calibrated | 0.0000 | 0.0000 | 0.0022 | 0 | 1 | PASS |
| AHL-DINO | S7 | fixed_cpu_threshold | 0.0282 | 0.0500 | 0.0018 | 1 | 1 | FAIL |
| AHL-DINO | S7 | self_calibrated | 0.0220 | 0.0000 | 0.0018 | 1 | 1 | FAIL |
| ADP-AHL-bridge-v2 | S7 | fixed_cpu_threshold | 0.0284 | 0.0500 | 0.0000 | 1 | 0 | FAIL |
| ADP-AHL-bridge-v2 | S7 | self_calibrated | 0.0000 | 0.0000 | 0.0000 | 1 | 0 | PASS |
| ADP-only-DINO | S8 | fixed_cpu_threshold | 0.0000 | 0.0000 | 0.0022 | 0 | 1 | PASS |
| ADP-only-DINO | S8 | self_calibrated | 0.0000 | 0.0000 | 0.0022 | 0 | 1 | PASS |
| AHL-DINO | S8 | fixed_cpu_threshold | 0.0000 | 0.0000 | 0.0000 | 0 | 1 | PASS |
| AHL-DINO | S8 | self_calibrated | 0.0256 | 0.0500 | 0.0000 | 0 | 1 | FAIL |
| ADP-AHL-bridge-v2 | S8 | fixed_cpu_threshold | 0.0000 | 0.0000 | 0.0024 | 0 | 1 | PASS |
| ADP-AHL-bridge-v2 | S8 | self_calibrated | 0.0000 | 0.0000 | 0.0024 | 0 | 1 | PASS |
