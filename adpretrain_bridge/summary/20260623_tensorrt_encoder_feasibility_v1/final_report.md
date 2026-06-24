# 20260623_tensorrt_encoder_feasibility_v1

## Scope

- Target: DINOv2-large encoder only.
- Input: fixed `[1,3,224,224]`.
- Output target: multi-scale encoder features.
- Explicitly not included: projector, ADP/AHL/bridge full-stage replacement, INT8, retraining, alpha changes, threshold policy changes.

## Initial Result

The initial TensorRT encoder diagnostic stopped at the dependency gate. ONNX export and TensorRT engine build were not attempted because the runtime environment did not provide the required TensorRT/ONNX tooling.

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
- TensorRT Python: unavailable at initial diagnostic
- `trtexec`: unavailable on `PATH`
- ONNX Python: unavailable at initial diagnostic
- ONNX Runtime Python: unavailable
- `nvcc`: `/usr/local/cuda/bin/nvcc`
- `nvcc --version`: CUDA `8.0`, stale relative to the active Torch CUDA runtime
- TF32 at initial diagnostic:
  - `torch.backends.cuda.matmul.allow_tf32 = false`
  - `torch.backends.cudnn.allow_tf32 = true`

Full raw inventory is in `env_inventory.json`.

## Initial Dependency Gate

Missing required components:

- `trtexec` executable not found on `PATH`
- TensorRT Python package not importable
- `onnx` Python package not importable

Per experiment rule, no dependencies were installed or modified during the initial diagnostic.

## Initial Export / Build / Equivalence / Latency

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
  - Failed before environment inventory because the `tools/` script did not include project root on `sys.path`.
  - Fixed by adding `/ghome/huangjd/code/detected/adpretrain_bridge` to `sys.path` before importing `common`.
- Retry job: `215342.Ghead`
  - Completed the dependency gate and wrote the initial summary.

## 2026-06-24 Install Attempt

A minimal pip install was attempted in the existing main conda environment:

```bash
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python -m pip install --only-binary=:all: 'onnx==1.17.0' 'tensorrt==8.6.0'
```

Installed packages:

- `onnx==1.17.0`
- `tensorrt==8.6.0`
- `protobuf==5.29.6`

Post-install check:

```json
{
  "onnx": "1.17.0",
  "protobuf": "5.29.6",
  "tensorrt_error": "ImportError('libcudnn.so.8: cannot open shared object file: No such file or directory')",
  "torch": "2.4.1+cu121",
  "torch_cuda": "12.1",
  "torchvision": "0.19.1+cu121",
  "trtexec_path": null
}
```

Interpretation:

- ONNX became importable.
- TensorRT Python was installed but not usable because `tensorrt==8.6.0` requires `libcudnn.so.8`.
- The main environment uses cuDNN 9 through the current PyTorch 2.4.1+cu121 stack.
- The TensorRT wheel did not provide `trtexec`.
- Downgrading or replacing cuDNN in the main ADP/AHL environment is not recommended.

Detailed install notes are in `tensorrt_install_attempt_20260624.md`.

## 2026-06-24 Rollback

Because the attempted TensorRT install did not provide a usable TensorRT runtime and did not provide `trtexec`, the newly installed packages were removed from the main ADP/AHL environment:

- removed `tensorrt==8.6.0`
- removed `onnx==1.17.0`
- removed `protobuf==5.29.6`

Post-rollback checks:

- torch import: OK (`2.4.1+cu121`)
- torchvision import: OK (`0.19.1+cu121`)
- TensorRT import: unavailable, as before
- ONNX import: unavailable, as before
- `trtexec`: unavailable, as before
- `pip check`: no broken requirements found

Post-rollback import details are in `post_rollback_import_check.json`.

## Final Conclusion

- ONNX export success: no.
- TensorRT FP32 engine build success: no.
- TensorRT FP16 engine build success: no.
- FP32 equivalence: not evaluated.
- FP16 acceleration: not evaluated.
- Main ADP/AHL conda environment status: preserved for normal ADP/AHL work after rollback.
- Recommendation: do not continue TensorRT setup inside `adpretrain_ahl_bridge`. If TensorRT is pursued, create a separate TensorRT probe environment with a compatible TensorRT/cuDNN stack and a real `trtexec` binary, then rerun this encoder-only diagnostic there.

## Artifacts

- Summary: `summary/20260623_tensorrt_encoder_feasibility_v1/`
- Runs: `runs/20260623_tensorrt_encoder_feasibility_v1/`
- Heavy artifacts: none produced.
- ONNX/engine files: none produced.
