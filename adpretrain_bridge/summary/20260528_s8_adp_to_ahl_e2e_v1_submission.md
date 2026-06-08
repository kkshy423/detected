# 20260528_s8_adp_to_ahl_e2e_v1 提交清单

- job_name: `s8_adp_ahl_e2e`
- launcher: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_20260528_s8_adp_to_ahl_e2e_v1.sh`
- pbs: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/20260528_s8_adp_to_ahl_e2e_v1.pbs`
- output_root: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260528_s8_adp_to_ahl_e2e_v1`
- mode: S8 ADP->AHL same-process end-to-end benchmark, calib+test, batch=1, warmup=5, no .npy write
- resource_policy: copied `pbs/pbs_test.pbs`; resource line set by user request to `#PBS -l nodes=1:gpus=2:b`; also changed `#PBS -N` and `SCRIPT_FILE_PATH`
- resource_note: current script is single-process and uses `cuda:0`; second allocated GPU is not used unless code is changed for multi-GPU.
- timing_scope: image_load_decode, transform_normalize, h2d_copy, encoder, adp_reference_match, residual, projector, compress, ahl_reference_attach, ahl_forward, score_postprocess, threshold, total

## Command

```bash
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python /ghome/huangjd/code/detected/adpretrain_bridge/benchmark_qm_xiepai_s8_adp_to_ahl_e2e.py --output-root /ghome/huangjd/code/detected/adpretrain_bridge/output/20260528_s8_adp_to_ahl_e2e_v1 --warmup 5 --n-ref 5 --device cuda:0
```