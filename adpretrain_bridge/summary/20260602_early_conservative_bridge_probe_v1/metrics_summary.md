# 20260602 Early Conservative Bridge Probe V1

- stages: S1-S4
- bridge score: `alpha * ADP_norm + (1-alpha) * AHL_norm`
- normalization: per-stage calib-normal robust median/MAD
- alpha grid: S1-S2 `0.90/0.80/0.70`; S3-S4 `0.80/0.70/0.60`
- threshold selection: calibration only; S1-S2 require `Recall > 0.90`; S3-S4 require `Recall > 0.85`
- test set is only for final reporting

| Stage | Method | alpha | threshold | P | R | F1 | Acc | AUC-ROC | AUC-PR | TP | FP | TN | FN | Total_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | ADP-only-DINO |  | 1.3744 | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9690 | 0.9151 | 69 | 39 | 141 | 1 | 118.6682 |
| S1 | AHL-DINO |  | 0.1634 | 0.5943 | 0.9000 | 0.7159 | 0.8000 | 0.9407 | 0.8926 | 63 | 43 | 137 | 7 | 134.9796 |
| S1 | early_bridge_candidate | 0.7000 | 0.4444 | 0.6796 | 1.0000 | 0.8092 | 0.8680 | 0.9780 | 0.9324 | 70 | 33 | 147 | 0 | 134.9796 |
| S2 | ADP-only-DINO |  | 1.3744 | 0.6389 | 0.9857 | 0.7753 | 0.8400 | 0.9690 | 0.9151 | 69 | 39 | 141 | 1 | 100.2370 |
| S2 | AHL-DINO |  | 0.1809 | 0.3681 | 0.9571 | 0.5317 | 0.5280 | 0.8650 | 0.8122 | 67 | 115 | 65 | 3 | 132.6986 |
| S2 | early_bridge_candidate | 0.7000 | 0.3711 | 0.6667 | 1.0000 | 0.8000 | 0.8600 | 0.9767 | 0.9318 | 70 | 35 | 145 | 0 | 132.6986 |
| S3 | ADP-only-DINO |  | 1.3926 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 | 98.8260 |
| S3 | AHL-DINO |  | 0.2899 | 0.6598 | 0.9143 | 0.7665 | 0.8440 | 0.9308 | 0.8905 | 64 | 33 | 147 | 6 | 135.3822 |
| S3 | early_bridge_candidate | 0.6000 | 0.5562 | 0.6931 | 1.0000 | 0.8187 | 0.8760 | 0.9792 | 0.9360 | 70 | 31 | 149 | 0 | 135.3822 |
| S4 | ADP-only-DINO |  | 1.3926 | 0.6635 | 0.9857 | 0.7931 | 0.8560 | 0.9690 | 0.9151 | 69 | 35 | 145 | 1 | 96.7071 |
| S4 | AHL-DINO |  | 0.4144 | 0.6436 | 0.9286 | 0.7602 | 0.8360 | 0.9392 | 0.9041 | 65 | 36 | 144 | 5 | 129.4798 |
| S4 | early_bridge_candidate | 0.6000 | 1.8615 | 0.8519 | 0.9857 | 0.9139 | 0.9480 | 0.9817 | 0.9437 | 69 | 12 | 168 | 1 | 129.4798 |

- CSV: `summary/20260602_early_conservative_bridge_probe_v1/metrics_summary.csv`