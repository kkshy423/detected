# 20260607_gpu_preprocess_equivalence_fix_v1 Submission

## PBS Jobs

| purpose | job_id | pbs | resource | status |
| --- | --- | --- | --- | --- |
| smoke, invalid calib slice | 210648.Ghead | `pbs/generated_submit/20260607_gpu_preprocess_equivalence_fix_v1_smoke.pbs` | `#PBS -l nodes=1:gpus=1:a` | failed, `--max-calib 4` contained no anomaly calibration sample |
| smoke, fixed calib | 210649.Ghead | `pbs/generated_submit/20260607_gpu_preprocess_equivalence_fix_v1_smoke.pbs` | `#PBS -l nodes=1:gpus=1:a` | completed, exit_status=0 |
| full S8 eval | 210650.Ghead | `pbs/generated_submit/20260607_gpu_preprocess_equivalence_fix_v1.pbs` | `#PBS -l nodes=1:gpus=1:a` | completed, exit_status=0 |

## Generated Files

- `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_20260607_gpu_preprocess_equivalence_fix_v1_smoke.sh`
- `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_20260607_gpu_preprocess_equivalence_fix_v1.sh`
- `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/20260607_gpu_preprocess_equivalence_fix_v1_smoke.pbs`
- `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/20260607_gpu_preprocess_equivalence_fix_v1.pbs`

## Output

- Full output root: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260607_gpu_preprocess_equivalence_fix_v1`
- Smoke output root: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260607_gpu_preprocess_equivalence_fix_v1_smoke`

## Note

This is eval-only. No ADP/AHL model body, threshold policy, split, or training artifact was changed.
