# S8 alpha probe - diagnostic, not main-table update

**Hard boundary:** diagnostic only. Do not update bridge v2 main table from this result. AHL score口径尚未对齐，本实验只看方向。

## Metrics

| alpha | backend | threshold | P | R | F1 | Acc | AUC-PR | TP | FP | TN | FN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | cpu_pil | 2.6908 | 0.8806 | 0.8429 | 0.8613 | 0.9240 | 0.9275 | 59 | 8 | 172 | 11 |
| 0.70 | gpu_tensor_uint8_aa_true | 3.0940 | 0.8545 | 0.6714 | 0.7520 | 0.8760 | 0.9178 | 47 | 8 | 172 | 23 |
| 0.50 | cpu_pil | 2.8303 | 0.8833 | 0.7571 | 0.8154 | 0.9040 | 0.9314 | 53 | 7 | 173 | 17 |
| 0.50 | gpu_tensor_uint8_aa_true | 2.5812 | 0.8793 | 0.7286 | 0.7969 | 0.8960 | 0.9234 | 51 | 7 | 173 | 19 |
| 0.35 | cpu_pil | 1.8046 | 0.8824 | 0.8571 | 0.8696 | 0.9280 | 0.9289 | 60 | 8 | 172 | 10 |
| 0.35 | gpu_tensor_uint8_aa_true | 1.6590 | 0.8806 | 0.8429 | 0.8613 | 0.9240 | 0.9220 | 59 | 8 | 172 | 11 |

## CPU/GPU gaps

| alpha | CPU F1 | GPU F1 | dF1 | CPU R | GPU R | dR | CPU AUPR | GPU AUPR | dAUPR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 0.8613 | 0.7520 | -0.1093 | 0.8429 | 0.6714 | -0.1714 | 0.9275 | 0.9178 | -0.0097 |
| 0.50 | 0.8154 | 0.7969 | -0.0185 | 0.7571 | 0.7286 | -0.0286 | 0.9314 | 0.9234 | -0.0079 |
| 0.35 | 0.8696 | 0.8613 | -0.0083 | 0.8571 | 0.8429 | -0.0143 | 0.9289 | 0.9220 | -0.0070 |

## Pre-registered checks

- P1 CPU alpha=0.35 F1 >= alpha=0.70 F1: PASS (0.8696 vs 0.8613).
- P2 GPU-CPU gap narrows at alpha=0.35 vs alpha=0.70 for both F1 and recall: PASS.

This is diagnostic only and not a main-table update.
