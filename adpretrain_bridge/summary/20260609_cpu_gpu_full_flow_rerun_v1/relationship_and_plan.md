# 20260609 CPU/GPU full-flow rerun v1 - relationship and plan

## Relationship and principle

- ADP-only-DINO is the training-free residual baseline. It uses normal reference images to build DINO projected residual references, compares each query patch against the reference bank, turns residuals into anomaly maps, and reports image-level anomaly scores. In this project ADP uses `num_ref=8`.
- AHL-DINO is a few-shot trainable detector on top of the ADP/DINO projected residual feature cache. It has its own reference setting and training loop; in this project AHL uses `n_ref=5`, `epochs=30`, `steps_per_epoch=20`, `batch_size=48`, `seed=20260517`.
- ADP-AHL bridge v2 is late score fusion, not another training procedure. For each stage, ADP and AHL scores are normalized using calibration normal scores only, then fused as `alpha * ADP_norm + (1 - alpha) * AHL_norm`. The alpha schedule is fixed: S1-S2=0.70, S3-S4=0.60, S5-S7=0.35, S8=0.70; S0 has no bridge row.
- Thresholds are selected only on calib. Test is only reported. The stage recall policy is S0-S2 best-F1 with recall>=0.90, S3-S6 best-F1 with recall>=0.85, and S7-S8 pure best-F1.

## What this rerun answers

The controlled variable is preprocessing:

- CPU version: CPU/PIL preprocessing, then the same DINO, ADP, AHL, and bridge logic.
- GPU version: `gpu_tensor_uint8_aa_true` preprocessing, then the same DINO, ADP, AHL, and bridge logic.

The decision is not based on one scalar. The useful checks are:

- ADP-only CPU vs GPU: should be near-identical because there is no training randomness.
- AHL CPU rerun vs v2: estimates few-shot training/run-to-run instability under the old CPU preprocessing.
- AHL GPU rerun vs CPU rerun: if this gap is not larger than CPU rerun vs v2, the apparent AHL drift is likely training instability rather than GPU preprocessing damage.
- Bridge CPU vs GPU: primary production-facing replacement signal, because bridge is the intended score-combination policy.

## PBS jobs

All GPU-using work is submitted through PBS. Login node work is limited to file inspection and pure CPU result aggregation.

Run scripts:

- `pbs/generated_run/run_20260609_cpu_pil_full_flow_rerun_v1.sh`
- `pbs/generated_run/run_20260609_gpu_uint8_full_flow_rerun_v1.sh`

Submit files:

- `pbs/generated_submit/pbs_20260609_cpu_pil_full_flow_rerun_v1.pbs`
- `pbs/generated_submit/pbs_20260609_gpu_uint8_full_flow_rerun_v1.pbs`

Expected outputs:

- `output/20260609_cpu_gpu_full_flow_rerun_v1/cpu_pil_adp_only`
- `output/20260609_cpu_gpu_full_flow_rerun_v1/cpu_pil_ahl`
- `output/20260609_cpu_gpu_full_flow_rerun_v1/gpu_uint8_adp_only`
- `output/20260609_cpu_gpu_full_flow_rerun_v1/gpu_uint8_ahl`
- `summary/20260609_cpu_gpu_full_flow_rerun_v1`
