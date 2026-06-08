# 20260604_s8_adp_to_ahl_e2e_preprocess_compare_v1 提交清单

- job_name: `s8_e2e_prepcmp`
- launcher: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_20260604_s8_adp_to_ahl_e2e_preprocess_compare_v1.sh`
- pbs: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/20260604_s8_adp_to_ahl_e2e_preprocess_compare_v1.pbs`
- resource_line: `#PBS -l nodes=1:gpus=1:a`
- qsub_job_id: `209808.Ghead`
- mode: S8 ADP->AHL single-image model-side E2E timing, sequential backend compare
- backend_1: `cpu_pil`
- backend_1_output: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260604_s8_adp_to_ahl_e2e_cpu_pil_v1`
- backend_2: `gpu_tensor`
- backend_2_output: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260604_s8_adp_to_ahl_e2e_gpu_tensor_v1`
- warmup: `5`
- n_ref: `5`
- device: `cuda:0`
- timing_scope: image_load_decode, transform_normalize, h2d_copy, pil_to_tensor_cpu, raw_h2d_copy, gpu_cast_scale, gpu_resize, gpu_center_crop, gpu_normalize, preprocess_total, ADP encoder, ADP reference matching, residual, projector, compress, AHL reference attach, AHL forward, score postprocess, threshold, gpu_inference_only, total

## Commands

```bash
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python /ghome/huangjd/code/detected/adpretrain_bridge/benchmark_qm_xiepai_s8_adp_to_ahl_e2e.py --preprocess-backend cpu_pil --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/20260604_s8_adp_to_ahl_e2e_cpu_pil_v1 --warmup 5 --n-ref 5 --device cuda:0
```

```bash
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python /ghome/huangjd/code/detected/adpretrain_bridge/benchmark_qm_xiepai_s8_adp_to_ahl_e2e.py --preprocess-backend gpu_tensor --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/20260604_s8_adp_to_ahl_e2e_gpu_tensor_v1 --warmup 5 --n-ref 5 --device cuda:0
```
