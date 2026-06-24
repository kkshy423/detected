# TensorRT install attempt report

Date: 2026-06-24

## Goal

Install the minimal TensorRT/ONNX tooling into the existing conda environment:

`/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge`

The goal was to make `20260623_tensorrt_encoder_feasibility_v1` runnable without changing the main ADP/AHL/bridge algorithms.

## Existing environment

- Python: 3.8.20
- Torch: 2.4.1+cu121
- Torch CUDA runtime: 12.1
- Torchvision: 0.19.1+cu121
- Pillow: 10.4.0
- Driver: 550.90.07
- NVIDIA-SMI CUDA: 12.4
- glibc: 2.17
- Existing cuDNN pip package: `nvidia-cudnn-cu12==9.1.0.70`

## Package-source feasibility

- Conda channel search was not usable in this environment. The default Anaconda channel returned HTTP 403, and override-channel searches timed out.
- Pip could resolve `onnx==1.17.0`.
- Pip could resolve `tensorrt==8.6.0` for Python 3.8 / manylinux_2_17.
- The `tensorrt==8.6.0` wheel contains TensorRT Python bindings and TensorRT shared libraries, but it does not contain `trtexec`.
- TensorRT 10.x pip component packages were not usable as a coherent set for this Python 3.8 environment: bindings were visible only up to 10.7.x, while matching libs were not resolvable.

## Installed

Installed with pip:

```bash
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python -m pip install --only-binary=:all: 'onnx==1.17.0' 'tensorrt==8.6.0'
```

Installed packages:

- `onnx==1.17.0`
- `tensorrt==8.6.0`
- `protobuf==5.29.6`

Before/after pip freeze snapshots:

- `pip_freeze_before_tensorrt_install.txt`
- `pip_freeze_after_tensorrt_install.txt`

## Verification result

Post-install import check:

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

- ONNX is now importable.
- TensorRT Python is installed but not usable because it requires `libcudnn.so.8`.
- The existing environment uses cuDNN 9 via `nvidia-cudnn-cu12==9.1.0.70`, required by the current PyTorch install.
- `trtexec` is still unavailable.

## Decision

This installation does not make the TensorRT encoder feasibility experiment runnable as specified, because the experiment requires both:

- TensorRT Python runtime
- `trtexec`

The current state satisfies neither:

- TensorRT import fails due to missing cuDNN 8.
- `trtexec` is not present.

Do not downgrade or replace the main environment cuDNN package to cuDNN 8. That risks breaking the currently working Torch 2.4.1+cu121 ADP/AHL experiment environment.

## Recommendation

Do not continue installing TensorRT inside the main `adpretrain_ahl_bridge` environment.

The controlled next step is to create a separate TensorRT-only environment, for example:

`/gdata1/huangjd/miniconda3/envs/adpretrain_trt_probe`

That environment can be allowed to use a TensorRT-compatible cuDNN/TensorRT stack without risking the main ADP/AHL workflow. If the separate environment can provide `trtexec`, TensorRT Python, ONNX, and Torch/DINO dependencies, rerun the same encoder-only diagnostic there.

## Current status

- Main environment was modified by adding ONNX/TensorRT/protobuf packages.
- ONNX is usable.
- TensorRT remains unusable.
- TensorRT feasibility PBS diagnostic was not rerun after this install attempt because the dependency gate would still fail on TensorRT Python import and `trtexec`.
