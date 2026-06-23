# PBS submission: 20260623_tensorrt_encoder_feasibility_v1

## Purpose

Diagnose TensorRT feasibility for DINOv2-large encoder only. No projector, no full-stage replacement, no INT8, no retraining, no threshold or alpha changes.

## Files

- Script: `tools/diagnose_tensorrt_encoder_feasibility.py`
- Run launcher: `pbs/generated_run/run_20260623_tensorrt_encoder_feasibility_v1.sh`
- PBS submit: `pbs/generated_submit/pbs_20260623_tensorrt_encoder_feasibility_v1.pbs`
- Summary dir: `summary/20260623_tensorrt_encoder_feasibility_v1/`
- Runs dir: `runs/20260623_tensorrt_encoder_feasibility_v1/`

## PBS template fields

- `#PBS -N trt_enc_feas_v1`
- `#PBS -l nodes=1:gpus=1:a` unchanged from template
- `SCRIPT_FILE_PATH=/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_20260623_tensorrt_encoder_feasibility_v1.sh`

## Command

```bash
qsub pbs/generated_submit/pbs_20260623_tensorrt_encoder_feasibility_v1.pbs
```

## Notes

If TensorRT Python or `trtexec` is missing, the script writes `env_inventory.json`, `missing_dependency_report.md`, and `final_report.md`, then exits without installing dependencies.

## Submitted

- Job ID: 215341.Ghead

## Retry 1

- Previous job: 215341.Ghead
- Previous result: failed before environment inventory because 	ools/ script did not include project root on sys.path; no TensorRT conclusion produced.
- Fix: add /ghome/huangjd/code/detected/adpretrain_bridge to sys.path before importing common.
- Resubmission command: qsub pbs/generated_submit/pbs_20260623_tensorrt_encoder_feasibility_v1.pbs

- Retry job ID: 215342.Ghead
