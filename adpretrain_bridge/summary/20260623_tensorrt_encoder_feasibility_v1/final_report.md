# 20260623_tensorrt_encoder_feasibility_v1

## Scope

- Target: DINOv2-large encoder only.
- Input: fixed `[1,3,224,224]`.
- Output target: multi-scale encoder features.
- Explicitly not included: projector, ADP/AHL/bridge full-stage replacement, INT8, retraining, alpha changes, threshold policy changes.

## Result

This diagnostic stopped at the dependency gate. ONNX export and TensorRT engine build were not attempted because the runtime environment does not currently provide the required TensorRT/ONNX tooling.

## Environment Inventory

- Python: `/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python`
- Platform: `Linux-3.10.0-1160.119.1.el7.x86_64-x86_64-with-glibc2.17`
- GPU: `NVIDIA A40`
- GPU capability: `8.6`
- Driver: `550.90.07`
- NVIDIA-SMI CUDA: `12.4`
- Torch: `2.4.1+cu121`
- Torch CUDA: `12.1`
- Torchvision: `0.19.1+cu121`
- Pillow: `10.4.0`
- cuDNN: `90100`
- TensorRT Python: unavailable
- `trtexec`: unavailable on `PATH`
- ONNX Python: unavailable
- ONNX Runtime Python: unavailable
- `nvcc`: `/usr/local/cuda/bin/nvcc`
- `nvcc --version`: CUDA `8.0`, which is stale relative to the active Torch CUDA runtime
- TF32 before export/build path:
  - `torch.backends.cuda.matmul.allow_tf32 = false`
  - `torch.backends.cudnn.allow_tf32 = true`

Full raw inventory is in `env_inventory.json`.

## Dependency Gate

Missing required components:

- `trtexec` executable not found on `PATH`
- TensorRT Python package not importable
- `onnx` Python package not importable

Per experiment rule, no dependencies were installed or modified.

## Export / Build / Equivalence / Latency

| Item | Status | Notes |
| --- | --- | --- |
| ONNX export | not attempted | blocked by missing ONNX Python package |
| TensorRT FP32 engine build | not attempted | blocked by missing `trtexec` and TensorRT Python |
| TensorRT FP16 engine build | not attempted | blocked by missing `trtexec` and TensorRT Python |
| FP32 feature equivalence | N/A | no engine/runtime available |
| FP16 feature diff | N/A | no engine/runtime available |
| Encoder latency | N/A | no TensorRT engine available |

## PBS Jobs

- First job: `215341.Ghead`
  - Failed before env inventory due to `tools/` script import path not including project root.
  - Fixed by adding `/ghome/huangjd/code/detected/adpretrain_bridge` to `sys.path` before importing `common`.
- Retry job: `215342.Ghead`
  - Completed the dependency gate and wrote this summary.

## Conclusion

- Export success: no, blocked before ONNX export.
- TensorRT FP32 build success: no, blocked by missing TensorRT/ONNX tooling.
- TensorRT FP16 build success: no, blocked by missing TensorRT/ONNX tooling.
- FP32 equivalence: not evaluated.
- FP16 acceleration: not evaluated.
- Recommendation: do not start encoder drop-in full-stage diagnostics on this server environment yet. The next decision point is whether to provision a TensorRT-capable environment with compatible `trtexec`, TensorRT Python bindings, and ONNX support. Only after that should this same encoder-only diagnostic be rerun.

## Artifacts

- Summary: `summary/20260623_tensorrt_encoder_feasibility_v1/`
- Runs: `runs/20260623_tensorrt_encoder_feasibility_v1/`
- Heavy artifacts: none produced.
- No ONNX or TensorRT engine was generated.
