# 20260607_gpu_preprocess_equivalence_fix_v1

## Scope

- Stage: `S8`
- Output root: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260607_gpu_preprocess_uint8_equiv_v1`
- Fixed CPU threshold source: `cpu_calib`
- Fixed CPU threshold: `0.575703978539`
- Threshold policy: `strategy_mild_stage_v2_1_safe`
- n_ref: `5`
- warmup: `5`
- Primary latency: `decoded_image_to_threshold_ms`; `image_load_decode_ms` is not part of the production-preloaded-image estimate.

## Fixed-Threshold Equivalence

| scenario | backend | test pred changed | score diff median | score diff P95 | Recall | F1 | preprocess median ms | accepted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| query_only_diff | cpu_pil | 0 | 0.000000 | 0.000000 | 0.7714 | 0.8372 | 19.38 | 0 |
| query_only_diff | gpu_tensor_current | 5 | 0.012184 | 0.040815 | 0.7857 | 0.8594 | 5.72 | 0 |
| query_only_diff | gpu_tensor_torchvision_aa_true | 5 | 0.012184 | 0.040815 | 0.7857 | 0.8594 | 5.47 | 0 |
| query_only_diff | gpu_tensor_torchvision_aa_false | 9 | 0.022383 | 0.081121 | 0.7000 | 0.8033 | 5.01 | 0 |
| query_only_diff | gpu_tensor_uint8_aa_true | 3 | 0.007369 | 0.023102 | 0.7571 | 0.8413 | 4.92 | 0 |
| ref_and_query_diff | cpu_pil | 0 | 0.000000 | 0.000000 | 0.7714 | 0.8372 | 19.38 | 0 |
| ref_and_query_diff | gpu_tensor_current | 7 | 0.012586 | 0.043090 | 0.7143 | 0.8197 | 5.72 | 0 |
| ref_and_query_diff | gpu_tensor_torchvision_aa_true | 7 | 0.012586 | 0.043090 | 0.7143 | 0.8197 | 5.47 | 0 |
| ref_and_query_diff | gpu_tensor_torchvision_aa_false | 11 | 0.025415 | 0.097374 | 0.6714 | 0.7833 | 5.01 | 0 |
| ref_and_query_diff | gpu_tensor_uint8_aa_true | 5 | 0.009103 | 0.033915 | 0.7571 | 0.8413 | 4.92 | 0 |

## Latency

| scenario | backend | preprocess median | tensor-to-threshold median | decoded-image-to-threshold median |
| --- | --- | ---: | ---: | ---: |
| query_only_diff | cpu_pil | 19.38 | 28.73 | 48.35 |
| query_only_diff | gpu_tensor_current | 5.72 | 28.69 | 35.36 |
| query_only_diff | gpu_tensor_torchvision_aa_true | 5.47 | 28.73 | 34.36 |
| query_only_diff | gpu_tensor_torchvision_aa_false | 5.01 | 28.71 | 33.70 |
| query_only_diff | gpu_tensor_uint8_aa_true | 4.92 | 28.71 | 33.62 |
| ref_and_query_diff | cpu_pil | 19.38 | 28.68 | 48.43 |
| ref_and_query_diff | gpu_tensor_current | 5.72 | 28.71 | 35.33 |
| ref_and_query_diff | gpu_tensor_torchvision_aa_true | 5.47 | 28.67 | 34.39 |
| ref_and_query_diff | gpu_tensor_torchvision_aa_false | 5.01 | 28.72 | 33.79 |
| ref_and_query_diff | gpu_tensor_uint8_aa_true | 4.92 | 28.67 | 33.64 |

## Tensor Difference

| backend | mean_abs_diff median | p95_abs_diff median | p99_abs_diff median | max_abs_diff P95 |
| --- | ---: | ---: | ---: | ---: |
| gpu_tensor_current | 0.00326620 | 0.00935064 | 0.04115092 | 0.34014845 |
| gpu_tensor_torchvision_aa_true | 0.00326620 | 0.00935064 | 0.04115092 | 0.34014845 |
| gpu_tensor_torchvision_aa_false | 0.01481789 | 0.05140562 | 0.25732720 | 2.44332807 |
| gpu_tensor_uint8_aa_true | 0.00081659 | 0.00000024 | 0.01750696 | 0.12430017 |

## Answers

1. Closest GPU backend: `gpu_tensor_uint8_aa_true` under `ref_and_query_diff` (test pred changed=5, score_abs_diff_P95=0.033915, F1=0.8413).
2. Query/reference source diagnosis:
- `gpu_tensor_current`: reference bank preprocess also increases the gap.
- `gpu_tensor_torchvision_aa_true`: reference bank preprocess also increases the gap.
- `gpu_tensor_torchvision_aa_false`: reference bank preprocess also increases the gap.
- `gpu_tensor_uint8_aa_true`: reference bank preprocess also increases the gap.
3. CPU-PIL replacement decision: No. No GPU backend met all acceptance criteria, so CPU-PIL should remain the production metric backend.
4. Production metric recommendation: keep `cpu_pil` as the production metric backend unless an accepted GPU backend appears in this summary.
