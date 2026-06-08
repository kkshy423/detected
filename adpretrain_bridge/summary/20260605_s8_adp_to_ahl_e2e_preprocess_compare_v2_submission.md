# 20260605_s8_adp_to_ahl_e2e_preprocess_compare_v2 提交清单

- previous_job: `209808.Ghead`
- previous_status: failed before result writing
- previous_failure: default `--stage-root` pointed to missing `stage_manifest.json`
- job_name: `s8_e2e_prepcmp2`
- qsub_job_id: `210110.Ghead`
- launcher: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_20260605_s8_adp_to_ahl_e2e_preprocess_compare_v2.sh`
- pbs: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/20260605_s8_adp_to_ahl_e2e_preprocess_compare_v2.pbs`
- resource_line: `#PBS -l nodes=1:gpus=1:a`
- stage_root: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/stage_roots_plain_dino_large_norm_val49/S8`
- backend_1: `cpu_pil`
- backend_1_output: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260605_s8_adp_to_ahl_e2e_cpu_pil_v2`
- backend_2: `gpu_tensor`
- backend_2_output: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260605_s8_adp_to_ahl_e2e_gpu_tensor_v2`
- warmup: `5`
- n_ref: `5`
- device: `cuda:0`
