# 20260623_tensorrt_encoder_feasibility_v1

## Scope

- Encoder only: DINOv2-large wrapper, fixed input [1,3,224,224].
- No projector, no full-stage replacement, no INT8, no retraining, no threshold/alpha change.

## Status

- ONNX export: success
- TensorRT FP32 build: success
- TensorRT FP16 build: success

## Latency

| backend | median_ms | mean_ms | p95_ms | min_ms | max_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| pytorch | 18.2894 | 18.2575 | 18.4773 | 18.0164 | 18.5356 |
| trt_fp32 | 7.0084 | 7.0229 | 7.0263 | 6.9682 | 8.4478 |
| trt_fp16 | 3.8948 | 3.9117 | 3.9057 | 3.8790 | 5.6152 |

## Feature Diff Summary

| backend | feature_idx | max_abs_max | mean_abs_mean | p95_abs_max | cosine_min | relative_l2_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trt_fp16 | 0 | 0.0167936 | 0.000374364 | 0.00100537 | 0.999983668 | 0.00575097 |
| trt_fp16 | 1 | 0.0222126 | 0.000525879 | 0.00133481 | 0.999962449 | 0.00866473 |
| trt_fp16 | 2 | 0.450294 | 0.00223301 | 0.00573531 | 0.999968529 | 0.00794176 |
| trt_fp16 | 3 | 5.27505 | 0.0232954 | 0.063484 | 0.999908209 | 0.0135531 |
| trt_fp32 | 0 | 0.00162621 | 9.74308e-05 | 0.000266458 | 0.999998987 | 0.00146739 |
| trt_fp32 | 1 | 0.00246787 | 0.000133942 | 0.000341386 | 0.999997973 | 0.00207809 |
| trt_fp32 | 2 | 0.0493507 | 0.000667766 | 0.00182934 | 0.999997675 | 0.00221799 |
| trt_fp32 | 3 | 0.385765 | 0.00794687 | 0.0212163 | 0.999992132 | 0.00419384 |

## Conclusion

- ONNX export succeeded.
- FP32 engine build succeeded.
- FP16 engine build succeeded.
- FP32 feature diff is not clearly equivalent; inspect feature_diff_summary.csv before any drop-in work.
- FP16 median latency speedup vs PyTorch encoder: 4.696x.
- Do not proceed to full-stage drop-in yet unless build/equivalence/latency issues are resolved.

## Artifacts

- runs: `runs/20260623_tensorrt_encoder_feasibility_v1`
- env inventory: `env_inventory.json`
- build results: `trt_build_results.json`
- latency: `encoder_latency.csv`
- feature diff detail: `feature_diff_detail.csv`
- feature diff summary: `feature_diff_summary.csv`
