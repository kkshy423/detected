# 20260607_cpu_gpu_uint8_full_stage_eval_v1 — CPU Cross-Check vs v2 main table

cpu_pil online (self_calibrated) vs summary/20260603 v2 main table. Drift if |diff|>0.02.

**Drift cells: 6/6** (0 expected if online ADP/AHL/bridge replicate the offline pipeline).

| method | stage | online F1 | v2 F1 | F1 diff | online R | v2 R | R diff | online AUC-PR | v2 AUC-PR | drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| ADP-only-DINO | S7 | 0.9474 | 0.8429 | 0.1045 | 0.9000 | 0.8429 | 0.0571 | 0.9976 | 0.9151 | DRIFT |
| AHL-DINO | S7 | 0.9000 | 0.7899 | 0.1101 | 0.9000 | 0.6714 | 0.2286 | 0.9817 | 0.9118 | DRIFT |
| ADP-AHL-bridge-v2 | S7 | 0.9474 | 0.8676 | 0.0797 | 0.9000 | 0.8429 | 0.0571 | 0.9907 | 0.9428 | DRIFT |
| ADP-only-DINO | S8 | 0.9474 | 0.8429 | 0.1045 | 0.9000 | 0.8429 | 0.0571 | 0.9976 | 0.9151 | DRIFT |
| AHL-DINO | S8 | 1.0000 | 0.8235 | 0.1765 | 1.0000 | 0.8000 | 0.2000 | 1.0000 | 0.8881 | DRIFT |
| ADP-AHL-bridge-v2 | S8 | 0.9474 | 0.8593 | 0.0881 | 0.9000 | 0.8286 | 0.0714 | 1.0000 | 0.9276 | DRIFT |
