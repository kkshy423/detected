# 20260530 Stage-Aware Bridge Stability Check V1

- bootstrap: 100 stratified calibration bootstraps per stage/method
- threshold policy: S5-S6 calibration best-F1 under `Recall>0.85`; S7-S8 calibration best-F1
- bridge alpha: S5-S7 `0.35`, S8 `0.70`
- normalization: bridge scores use full calibration-normal robust median/MAD, then bootstrap only reselects thresholds
- test set is fixed and never used for threshold selection

| Stage | Method | F1 mean | F1 std | F1 p05 | F1 p50 | F1 p95 | P mean | R mean | Acc mean | FP mean | FN mean | win_rate_vs_ADP | win_rate_vs_AHL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S5 | ADP-only-DINO | 0.8058 | 0.0481 | 0.7582 | 0.7931 | 0.8571 | 0.6990 | 0.9634 | 0.8661 | 30.9100 | 2.5600 |  | 0.7400 |
| S5 | AHL-DINO | 0.7801 | 0.0236 | 0.7602 | 0.7888 | 0.8105 | 0.6937 | 0.8997 | 0.8568 | 28.7900 | 7.0200 | 0.2600 |  |
| S5 | bridge_alpha_0.35 | 0.8815 | 0.0184 | 0.8475 | 0.8936 | 0.8951 | 0.8603 | 0.9086 | 0.9314 | 10.7600 | 6.4000 | 1.0000 | 1.0000 |
| S6 | ADP-only-DINO | 0.8094 | 0.0302 | 0.7582 | 0.7931 | 0.8464 | 0.7007 | 0.9660 | 0.8710 | 29.8600 | 2.3800 |  | 0.4900 |
| S6 | AHL-DINO | 0.8042 | 0.0426 | 0.6733 | 0.8098 | 0.8429 | 0.7280 | 0.9137 | 0.8722 | 25.9100 | 6.0400 | 0.4800 |  |
| S6 | bridge_alpha_0.35 | 0.8883 | 0.0156 | 0.8587 | 0.8896 | 0.9150 | 0.8631 | 0.9201 | 0.9355 | 10.5400 | 5.5900 | 0.9700 | 0.9900 |
| S7 | ADP-only-DINO | 0.8099 | 0.0535 | 0.7179 | 0.8406 | 0.8511 | 0.8300 | 0.8081 | 0.8961 | 12.5400 | 13.4300 |  | 0.8000 |
| S7 | AHL-DINO | 0.7720 | 0.0393 | 0.7027 | 0.7899 | 0.8406 | 0.9354 | 0.6641 | 0.8917 | 3.5600 | 23.5100 | 0.2000 |  |
| S7 | bridge_alpha_0.35 | 0.8707 | 0.0213 | 0.8567 | 0.8676 | 0.8919 | 0.8848 | 0.8607 | 0.9290 | 8.0100 | 9.7500 | 0.9700 | 1.0000 |
| S8 | ADP-only-DINO | 0.8040 | 0.0571 | 0.7283 | 0.8382 | 0.8464 | 0.8298 | 0.7989 | 0.8936 | 12.5100 | 14.0800 |  | 0.6600 |
| S8 | AHL-DINO | 0.8037 | 0.0308 | 0.7521 | 0.8148 | 0.8235 | 0.8693 | 0.7536 | 0.8981 | 8.2300 | 17.2500 | 0.3400 |  |
| S8 | bridge_alpha_0.70 | 0.8456 | 0.0221 | 0.8000 | 0.8507 | 0.8593 | 0.8668 | 0.8340 | 0.9144 | 9.7800 | 11.6200 | 0.8400 | 0.9200 |

## Decision

- S5: bridge mainline confirmed; bridge win_rate_vs_ADP=1.0000, R mean=0.9086.
- S6: bridge mainline confirmed; bridge win_rate_vs_ADP=0.9700, R mean=0.9201.
- S7: bridge mainline confirmed; bridge win_rate_vs_ADP=0.9700, R mean=0.8607.
- S8: use bridge-alpha70 in main table; bridge win_rate_vs_ADP=0.8400, R mean=0.8340.