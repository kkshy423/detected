# 20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1 Submission

## PBS Jobs

| purpose | job_id | pbs | resource | status |
| --- | --- | --- | --- | --- |
| 1-epoch smoke | 210651.Ghead | `pbs/generated_submit/20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1_smoke.pbs` | `#PBS -l nodes=1:gpus=1:a` | completed, exit_status=0 |
| full S8 GPU-preprocess AHL retrain | 210652.Ghead | `pbs/generated_submit/20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1.pbs` | `#PBS -l nodes=1:gpus=1:a` | completed, exit_status=0 |
| online timing summary rerun after phase-merge fix | 210661.Ghead | `pbs/generated_submit/20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1.pbs` | `#PBS -l nodes=1:gpus=1:a` | completed, exit_status=0 |

## Generated Files

- `/ghome/huangjd/code/detected/adpretrain_bridge/run_s8_gpu_preprocess_ahl_retrain_smoke.py`
- `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1_smoke.sh`
- `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1.sh`
- `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1_smoke.pbs`
- `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1.pbs`

## Output

- Full output root: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1`
- Full summary root: `/ghome/huangjd/code/detected/adpretrain_bridge/summary/20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1`
- Smoke output root: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1_smoke`
- Smoke summary root: `/ghome/huangjd/code/detected/adpretrain_bridge/summary/20260607_s8_gpu_preprocess_ahl_retrain_smoke_v1_smoke`

## Note

This experiment changed only the preprocessing backend for feature export/online scoring and retrained S8 AHL weights. ADP backbone, ADP projector, AHL structure, AHL loss, split, and threshold policy were unchanged.
