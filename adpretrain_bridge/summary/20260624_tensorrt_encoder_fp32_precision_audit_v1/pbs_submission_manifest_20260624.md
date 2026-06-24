# PBS submission manifest: 20260624_tensorrt_encoder_fp32_precision_audit_v1

## Generated files

- run script: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_20260624_tensorrt_encoder_fp32_precision_audit_v1.sh`
- PBS file: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/pbs_20260624_tensorrt_encoder_fp32_precision_audit_v1.pbs`
- stdout: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/20260624_tensorrt_encoder_fp32_precision_audit_v1.out`
- stderr: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/20260624_tensorrt_encoder_fp32_precision_audit_v1.err`

## Environment

- conda env: `/gdata1/huangjd/miniconda3/envs/adpretrain_trt_probe`
- extra TensorRT import library path: `/gdata1/huangjd/tensorrt_probe_py38_extra_libs/nvidia/cudnn/lib`
- GPU task policy: submitted through PBS only.

## Audit scope

- fixed sample count: `20`
- pytorch baseline: `TF32 off`
- engine variants: `default`, `disable_tf32`, `disable_tf32 + stronger_fp32_constraints if supported`
- stop condition: `do not enter full-stage drop-in in this round`

## Command

```bash
qsub pbs/generated_submit/pbs_20260624_tensorrt_encoder_fp32_precision_audit_v1.pbs
```

## Submitted jobs

- 2026-06-24 16:24:23 CST job_id: `215582.Ghead`, pbs: `pbs/generated_submit/pbs_20260624_tensorrt_encoder_fp32_precision_audit_v1.pbs`
