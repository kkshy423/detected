# 20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1

## Setup

- Stage: `S8`
- GPU preprocessing backend: `gpu_tensor_torchvision_aa_true`
- Threshold policy: `strategy_mild_stage_v2_1_safe`
- CPU AHL weights: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260529_ahl_dino_large_180_70_val49_stage_v1/stages/S8/ahl/models_qiumianxiepai_ctest.pkl`
- GPU retrain weights: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1/stages/S8/ahl/models_qiumianxiepai_ctest.pkl`
- GPU cache root: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/plain_dino_large_norm_gpu_tensor_aa_true_s8_retrain_smoke_v1`
- Export status: `reused`
- Online n_ref: `5`; ADP export num_ref: `8`

## Metrics

| Group | threshold | P | R | F1 | Acc | AUC-ROC | AUC-PR | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu_baseline | 0.575704 | 0.9153 | 0.7714 | 0.8372 | 0.9160 | 0.9612 | 0.9167 | 54 | 5 | 175 | 16 |
| gpu_no_retrain | 0.582367 | 0.9600 | 0.6857 | 0.8000 | 0.9040 | 0.9630 | 0.9097 | 48 | 2 | 178 | 22 |
| gpu_retrain | 0.619126 | 0.9245 | 0.7000 | 0.7967 | 0.9000 | 0.9462 | 0.8951 | 49 | 4 | 176 | 21 |

## Latency

| Group | decoded median | decoded P95 | preprocess median | tensor-to-threshold median |
| --- | ---: | ---: | ---: | ---: |
| cpu_baseline | 41.33 | 47.06 | 13.59 | 27.82 |
| gpu_no_retrain | 31.27 | 32.67 | 3.51 | 27.84 |
| gpu_retrain | 31.28 | 32.70 | 3.51 | 27.80 |

## Changed Samples Vs CPU Baseline

- Changed rows: `17`

## Decision

- GPU retrain F1 gap vs CPU baseline: `0.0405`
- GPU retrain AUC-PR gap vs CPU baseline: `0.0216`
- GPU retrain Recall drop vs CPU baseline: `0.0714`
- GPU retrain decoded-image-to-threshold median: `31.28 ms`
- Decision: `stop_gpu_preprocess_replacement_route`
- Interpretation: GPU retrain still does not recover CPU-PIL baseline closely enough; keep CPU-PIL as production metric/backend.
