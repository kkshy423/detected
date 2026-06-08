# 20260607_gpu_preprocess_equivalence_fix_v1

## Scope

- Stage: `S8`
- Output root: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260607_gpu_preprocess_equivalence_fix_v1`
- Fixed CPU threshold source: `cpu_calib`
- Fixed CPU threshold: `0.575703978539`
- Threshold policy: `strategy_mild_stage_v2_1_safe`
- n_ref: `5`
- warmup: `5`
- Primary latency: `decoded_image_to_threshold_ms`; `image_load_decode_ms` is not part of the production-preloaded-image estimate.

## Fixed-Threshold Equivalence

| scenario | backend | test pred changed | score diff median | score diff P95 | Recall | F1 | preprocess median ms | accepted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| query_only_diff | cpu_pil | 0 | 0.000000 | 0.000000 | 0.7714 | 0.8372 | 24.64 | 0 |
| query_only_diff | gpu_tensor_current | 5 | 0.012184 | 0.040815 | 0.7857 | 0.8594 | 5.87 | 0 |
| query_only_diff | gpu_tensor_torchvision_aa_true | 5 | 0.012184 | 0.040815 | 0.7857 | 0.8594 | 5.53 | 0 |
| query_only_diff | gpu_tensor_torchvision_aa_false | 9 | 0.022383 | 0.081121 | 0.7000 | 0.8033 | 5.33 | 0 |
| ref_and_query_diff | cpu_pil | 0 | 0.000000 | 0.000000 | 0.7714 | 0.8372 | 24.64 | 0 |
| ref_and_query_diff | gpu_tensor_current | 7 | 0.012586 | 0.043090 | 0.7143 | 0.8197 | 5.87 | 0 |
| ref_and_query_diff | gpu_tensor_torchvision_aa_true | 7 | 0.012586 | 0.043090 | 0.7143 | 0.8197 | 5.53 | 0 |
| ref_and_query_diff | gpu_tensor_torchvision_aa_false | 11 | 0.025415 | 0.097374 | 0.6714 | 0.7833 | 5.33 | 0 |

## Latency

| scenario | backend | preprocess median | tensor-to-threshold median | decoded-image-to-threshold median |
| --- | --- | ---: | ---: | ---: |
| query_only_diff | cpu_pil | 24.64 | 28.60 | 54.69 |
| query_only_diff | gpu_tensor_current | 5.87 | 28.61 | 34.69 |
| query_only_diff | gpu_tensor_torchvision_aa_true | 5.53 | 28.60 | 34.11 |
| query_only_diff | gpu_tensor_torchvision_aa_false | 5.33 | 28.62 | 33.96 |
| ref_and_query_diff | cpu_pil | 24.64 | 28.57 | 54.87 |
| ref_and_query_diff | gpu_tensor_current | 5.87 | 28.63 | 34.68 |
| ref_and_query_diff | gpu_tensor_torchvision_aa_true | 5.53 | 28.58 | 34.09 |
| ref_and_query_diff | gpu_tensor_torchvision_aa_false | 5.33 | 28.62 | 33.91 |

## Tensor Difference

| backend | mean_abs_diff median | p95_abs_diff median | p99_abs_diff median | max_abs_diff P95 |
| --- | ---: | ---: | ---: | ---: |
| gpu_tensor_current | 0.00326620 | 0.00935064 | 0.04115092 | 0.34014845 |
| gpu_tensor_torchvision_aa_true | 0.00326620 | 0.00935064 | 0.04115092 | 0.34014845 |
| gpu_tensor_torchvision_aa_false | 0.01481789 | 0.05140562 | 0.25732720 | 2.44332807 |

## Answers

1. Closest GPU backend: `gpu_tensor_current` under `ref_and_query_diff` (test pred changed=7, score_abs_diff_P95=0.043090, F1=0.8197).
2. Query/reference source diagnosis:
- `gpu_tensor_current`: reference bank preprocess also increases the gap.
- `gpu_tensor_torchvision_aa_true`: reference bank preprocess also increases the gap.
- `gpu_tensor_torchvision_aa_false`: reference bank preprocess also increases the gap.
3. CPU-PIL replacement decision: No. No GPU backend met all acceptance criteria, so CPU-PIL should remain the production metric backend.
4. Production metric recommendation: keep `cpu_pil` as the production metric backend unless an accepted GPU backend appears in this summary.
