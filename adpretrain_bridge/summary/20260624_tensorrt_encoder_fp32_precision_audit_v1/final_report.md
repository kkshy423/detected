# 20260624_tensorrt_encoder_fp32_precision_audit_v1

## Scope

- Encoder only.
- No projector, no full-stage replacement, no FP16 drop-in, no main-env change.
- PyTorch baseline uses TF32 off.

## Engine Config

- trt_fp32_default: tf32_enabled=True, flags=['TF32'], returncode=0
- trt_fp32_tf32_disabled: tf32_enabled=False, flags=[], returncode=0
- trt_fp32_tf32_disabled_strict: tf32_enabled=False, flags=['OBEY_PRECISION_CONSTRAINTS'], returncode=0

## Latency

| backend | median_ms | mean_ms | p95_ms | min_ms | max_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| pytorch_tf32_off | 18.3162 | 18.3053 | 18.5334 | 18.0453 | 18.5600 |
| trt_fp32_default | 7.0073 | 7.0116 | 7.0493 | 6.9507 | 7.3080 |
| trt_fp32_tf32_disabled | 13.3797 | 13.3863 | 13.4645 | 12.6531 | 15.7487 |
| trt_fp32_tf32_disabled_strict | 13.4005 | 16.2524 | 13.5735 | 12.6009 | 286.8038 |

## Feature Diff Summary

| backend | feature_idx | max_abs_max | mean_abs_mean | p95_abs_max | cosine_min | relative_l2_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trt_fp32_default | 0 | 0.00199968 | 5.42193e-05 | 0.000181464 | 0.999998927 | 0.00147851 |
| trt_fp32_default | 1 | 0.00587794 | 0.00012598 | 0.000328741 | 0.999998331 | 0.00183046 |
| trt_fp32_default | 2 | 0.0764389 | 0.000472365 | 0.00132717 | 0.999998510 | 0.0017388 |
| trt_fp32_default | 3 | 0.301048 | 0.00601177 | 0.017725 | 0.999993145 | 0.00380168 |
| trt_fp32_tf32_disabled | 0 | 5.92694e-06 | 6.40577e-08 | 2.05553e-07 | 0.999999940 | 1.78816e-06 |
| trt_fp32_tf32_disabled | 1 | 1.05798e-05 | 1.83808e-07 | 5.21541e-07 | 0.999999940 | 2.8292e-06 |
| trt_fp32_tf32_disabled | 2 | 9.15527e-05 | 5.84378e-07 | 1.60933e-06 | 0.999999881 | 2.07348e-06 |
| trt_fp32_tf32_disabled | 3 | 0.000898361 | 5.49859e-06 | 1.5378e-05 | 0.999999881 | 3.57348e-06 |
| trt_fp32_tf32_disabled_strict | 0 | 5.92694e-06 | 6.40577e-08 | 2.05553e-07 | 0.999999940 | 1.78816e-06 |
| trt_fp32_tf32_disabled_strict | 1 | 1.05798e-05 | 1.83808e-07 | 5.21541e-07 | 0.999999940 | 2.8292e-06 |
| trt_fp32_tf32_disabled_strict | 2 | 9.15527e-05 | 5.84378e-07 | 1.60933e-06 | 0.999999881 | 2.07348e-06 |
| trt_fp32_tf32_disabled_strict | 3 | 0.000898361 | 5.49859e-06 | 1.5378e-05 | 0.999999881 | 3.57348e-06 |

## Conclusion

- disable_tf32 deepest-feature relative_l2_max changed from 0.00380168 to 3.57348e-06.
- disable_tf32 deepest-feature cosine_min changed from 0.999993145 to 0.999999881.
- disable_tf32 median latency changed from 7.0073 ms to 13.3797 ms.
- TF32 root-cause hypothesis supported in this audit.
- stronger FP32 constraints deepest-feature relative_l2_max = 3.57348e-06, cosine_min = 0.999999881.
- Stop here. Do not enter full-stage drop-in from this audit.

## Artifacts

- `build_config_manifest.json`
- `fp32_precision_audit_latency.csv`
- `fp32_precision_audit_feature_diff.csv`
