# TensorRT Probe Environment Notes

## Purpose

This note records the isolated TensorRT probe environment used by
`20260623_tensorrt_encoder_feasibility_v1`.

## Environment

- conda env: `/gdata1/huangjd/miniconda3/envs/adpretrain_trt_probe`
- base clone source: `/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge`
- TensorRT python: `8.6.0`
- ONNX python: `1.17.0`
- extra cuDNN8 library path:
  `/gdata1/huangjd/tensorrt_probe_py38_extra_libs/nvidia/cudnn/lib`

## Rationale

- Main environment stays unchanged because PyTorch 2.4.1 uses cuDNN 9.
- TensorRT 8.6 python binding requires `libcudnn.so.8`.
- To avoid breaking the main project stack, cuDNN 8 is kept in an isolated
  external directory and only injected through `LD_LIBRARY_PATH` inside the
  dedicated PBS run script.

## Current Scope

- This environment is validated for encoder-only TensorRT probing.
- It is not yet approved for full ADP/AHL/bridge production runs.
- `trtexec` is still unavailable on this cluster path; current build path uses
  the TensorRT Python API.

## Re-run

Use the generated PBS launcher:

```bash
qsub pbs/generated_submit/pbs_20260623_tensorrt_encoder_feasibility_v1_probe_env.pbs
```
