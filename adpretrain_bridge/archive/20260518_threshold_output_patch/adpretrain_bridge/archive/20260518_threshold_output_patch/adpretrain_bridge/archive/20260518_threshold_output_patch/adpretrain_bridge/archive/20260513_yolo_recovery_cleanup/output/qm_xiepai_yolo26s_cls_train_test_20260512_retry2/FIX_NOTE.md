# YOLO26s-cls Retry2 Dependency Fix Note

## Failure

Retry1 jobs failed during `ensure_ultralytics()` with:

```text
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

## Root Cause

The previous dependency bootstrap installed `ultralytics` into `/ghome/huangjd/.local/lib/python3.8/site-packages`. That installation imported the non-headless OpenCV package, which requires `libGL.so.1`. The PBS Docker image used by this project does not provide `libGL.so.1`, so `import cv2` failed before YOLO training started.

## Fix

`run_qm_xiepai_yolo26s_cls_stage.py` now uses a project-private headless dependency directory:

```text
/gdata1/huangjd/data/yolo26s_cls_py38_headless_deps_20260512
```

The script installs and imports `opencv-python-headless`, `ultralytics-thop`, and `ultralytics` from that directory with priority over user site-packages. The retry2 PBS launchers also set:

```bash
export PYTHONNOUSERSITE=1
export YOLO_PYTHON_DEPS_DIR=/gdata1/huangjd/data/yolo26s_cls_py38_headless_deps_20260512
```

A preflight import confirmed:

```text
cv2 -> /gdata1/huangjd/data/yolo26s_cls_py38_headless_deps_20260512/cv2/__init__.py
ultralytics -> /gdata1/huangjd/data/yolo26s_cls_py38_headless_deps_20260512/ultralytics/__init__.py
```

## Retry Output

```text
/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260512_retry2
```

S0 is skipped because supervised binary YOLO-cls has no anomaly sample. S1-S8 were resubmitted through PBS.