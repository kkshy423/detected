# 20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1

## Setup

- Stage: `S8`
- GPU preprocessing backend: `gpu_tensor_torchvision_aa_true`
- Threshold policy: `strategy_mild_stage_v2_1_safe`
- CPU AHL weights: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260529_ahl_dino_large_180_70_val49_stage_v1/stages/S8/ahl/models_qiumianxiepai_ctest.pkl`
- GPU retrain weights: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1_smoke/stages/S8/ahl/models_qiumianxiepai_ctest.pkl`
- GPU cache root: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/plain_dino_large_norm_gpu_tensor_aa_true_s8_retrain_smoke_v1_smoke`
- Export status: `exported`
- Online n_ref: `5`; ADP export num_ref: `8`

## Metrics

| Group | threshold | P | R | F1 | Acc | AUC-ROC | AUC-PR | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu_baseline | 0.575704 | 0.9153 | 0.7714 | 0.8372 | 0.9160 | 0.9612 | 0.9167 | 54 | 5 | 175 | 16 |
| gpu_no_retrain | 0.582367 | 0.9600 | 0.6857 | 0.8000 | 0.9040 | 0.9630 | 0.9097 | 48 | 2 | 178 | 22 |
| gpu_retrain | 0.426497 | 0.8947 | 0.4857 | 0.6296 | 0.8400 | 0.8710 | 0.7925 | 34 | 4 | 176 | 36 |

## Latency

| Group | decoded median | decoded P95 | preprocess median | tensor-to-threshold median |
| --- | ---: | ---: | ---: | ---: |
| cpu_baseline | 41.73 | 47.77 | 13.92 | 0.00 |
| gpu_no_retrain | 31.67 | 34.13 | 3.87 | 0.00 |
| gpu_retrain | 31.76 | 34.04 | 3.87 | 0.00 |

## Changed Samples Vs CPU Baseline

- Changed rows: `34`

## Decision

- GPU retrain F1 gap vs CPU baseline: `0.2076`
- GPU retrain AUC-PR gap vs CPU baseline: `0.1242`
- GPU retrain Recall drop vs CPU baseline: `0.2857`
- GPU retrain decoded-image-to-threshold median: `31.76 ms`
- Decision: `stop_gpu_preprocess_replacement_route`
- Interpretation: GPU retrain still does not recover CPU-PIL baseline closely enough; keep CPU-PIL as production metric/backend.
