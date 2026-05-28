# YOLO26s-cls Retry1 Fix Note

## Failure

The original YOLO26s-cls PBS jobs failed before training. Example from S4:

```text
FileExistsError: [Errno 17] File exists: '/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260512/stages/S4'
```

## Root Cause

The PBS launcher created the stage log directory before starting Python:

```bash
mkdir -p .../output/qm_xiepai_yolo26s_cls_train_test_20260512/stages/S4/logs
```

This also created the parent stage directory. Then `run_qm_xiepai_yolo26s_cls_stage.py` tried to create the same stage directory with `exist_ok=False`, so it exited before YOLO training started.

## Fix

`run_qm_xiepai_yolo26s_cls_stage.py` now allows an existing stage directory, but refuses to overwrite an existing successful `metrics.json`.

The retry uses a new output root to preserve the failed first run:

```text
/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260512_retry1
```

S0 is marked as skipped because binary supervised YOLO-cls has no anomaly sample at S0. S1-S8 were resubmitted through PBS.