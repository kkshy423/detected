# 20260606 S8 ADP->AHL E2E preprocess compare v2 results

## Scope

- Job: `210110.Ghead`
- Stage: `S8`
- Resource: `#PBS -l nodes=1:gpus=1:a`
- CPU backend: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260605_s8_adp_to_ahl_e2e_cpu_pil_v2`
- GPU tensor backend: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260605_s8_adp_to_ahl_e2e_gpu_tensor_v2`
- Steady samples: `245` test samples after `warmup=5`
- Note: `gpu_tensor` keeps PIL decode on CPU, then moves raw uint8 tensor to GPU for cast/resize/crop/normalize.

## Main Latency Comparison

| Metric | cpu_pil median | gpu_tensor median | change | cpu_pil mean | gpu_tensor mean | change | cpu_pil P95 | gpu_tensor P95 | change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total_ms | 214.02 | 93.95 | -56.10% | 303.86 | 90.24 | -70.30% | 621.21 | 117.74 | -81.05% |
| preprocess_total_ms | 167.57 | 28.41 | -83.05% | 252.21 | 38.02 | -84.92% | 572.77 | 75.43 | -86.83% |
| gpu_inference_only_ms | 32.66 | 38.79 | +18.77% | 51.32 | 51.75 | +0.84% | 110.83 | 85.45 | -22.90% |

## Preprocess Breakdown

| Phase | cpu_pil median | gpu_tensor median | cpu_pil mean | gpu_tensor mean | cpu_pil P95 | gpu_tensor P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| image_load_decode_ms | 91.11 | 9.73 | 92.94 | 10.09 | 125.99 | 12.81 |
| transform_normalize_ms | 55.30 | 16.48 | 158.19 | 25.70 | 482.94 | 64.13 |
| h2d_copy_ms | 0.30 | 1.20 | 1.09 | 2.23 | 0.43 | 1.76 |
| pil_to_tensor_cpu_ms | 0.00 | 15.10 | 0.00 | 22.87 | 0.00 | 61.48 |
| raw_h2d_copy_ms | 0.00 | 1.20 | 0.00 | 2.23 | 0.00 | 1.76 |
| gpu_cast_scale_ms | 0.00 | 0.27 | 0.00 | 0.94 | 0.00 | 0.43 |
| gpu_resize_ms | 0.00 | 0.19 | 0.00 | 0.70 | 0.00 | 0.27 |
| gpu_center_crop_ms | 0.00 | 0.07 | 0.00 | 0.84 | 0.00 | 0.15 |
| gpu_normalize_ms | 0.00 | 0.10 | 0.00 | 0.35 | 0.00 | 0.19 |

## Model-Side Breakdown

| Phase | cpu_pil median | gpu_tensor median | cpu_pil mean | gpu_tensor mean | cpu_pil P95 | gpu_tensor P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| encoder_ms | 19.40 | 22.58 | 37.98 | 36.66 | 96.92 | 68.52 |
| adp_reference_match_ms | 0.87 | 1.03 | 0.97 | 1.39 | 1.54 | 1.57 |
| residual_ms | 0.08 | 0.10 | 0.09 | 0.11 | 0.14 | 0.15 |
| projector_ms | 7.72 | 7.93 | 8.61 | 9.52 | 9.05 | 9.93 |
| compress_ms | 0.88 | 0.91 | 1.17 | 0.92 | 0.99 | 1.00 |
| ahl_forward_ms | 2.02 | 2.33 | 2.24 | 2.87 | 3.43 | 3.91 |

## Metric / Numerical Consistency Check

| Backend | Acc | P | R | F1 | AUC-ROC | AUC-PR | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu_pil | 0.8600 | 0.7778 | 0.7000 | 0.7368 | 0.8883 | 0.8410 | 49 | 14 | 166 | 21 |
| gpu_tensor | 0.8480 | 0.8077 | 0.6000 | 0.6885 | 0.8498 | 0.7987 | 42 | 10 | 170 | 28 |

| Consistency item | Value |
| --- | ---: |
| common rows | 250 |
| prediction mismatches | 29 |
| score abs diff mean | 0.026229 |
| score abs diff P50 | 0.021735 |
| score abs diff P95 | 0.063928 |
| score abs diff max | 0.138050 |

## Conclusion

The `gpu_tensor` path substantially reduces measured latency, especially `preprocess_total_ms` and `total_ms`. However, it is not numerically equivalent to the legacy `cpu_pil` path: 29/250 test predictions changed, recall dropped from 0.7000 to 0.6000, F1 dropped from 0.7368 to 0.6885, and AUC-PR dropped from 0.8410 to 0.7987.

Therefore this run proves the engineering speed-up direction is useful, but the current GPU tensor preprocessing path cannot replace the production timing baseline yet. The next engineering step should be numerical alignment of resize/crop/normalization with the legacy PIL/torchvision path, or a deliberate decision to recalibrate thresholds for the new preprocessing backend.

Additional caution: `gpu_tensor` ran after `cpu_pil`, so `image_load_decode_ms` likely benefited from OS file cache warming. The more reliable acceleration signal is the transform section: `transform_normalize_ms` median improved from 55.30 ms to 16.48 ms, while GPU tensor operations themselves are sub-ms median; the remaining median bottleneck inside the GPU path is `pil_to_tensor_cpu_ms` at 15.10 ms.
