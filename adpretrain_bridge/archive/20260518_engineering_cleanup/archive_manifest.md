# 20260518 Engineering Cleanup Manifest

created_at: 2026-05-18T14:18:11

## Actions

- move: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_ahl_plain_6_1_val_threshold_20260517` -> `/ghome/huangjd/code/detected/adpretrain_bridge/archive/20260518_engineering_cleanup/failed_outputs/qm_xiepai_ahl_plain_6_1_val_threshold_20260517_failed_py38`; reason: all stages failed before python3.8 compatibility patch
- move: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_6_1_val_threshold_20260517/stages/S4` -> `/ghome/huangjd/code/detected/adpretrain_bridge/archive/20260518_engineering_cleanup/failed_outputs/qm_xiepai_yolo26s_cls_6_1_val_threshold_20260517_S4_failed_amp`; reason: S4 failed during ultralytics AMP check
- legacy-copy: `/ghome/huangjd/code/detected/adpretrain_bridge/prepare_qm_xiepai_yolo26s_cls_stages.py` -> `/ghome/huangjd/code/detected/adpretrain_bridge/archive/20260518_engineering_cleanup/legacy_script_copies/prepare_qm_xiepai_yolo26s_cls_stages.py`; reason: legacy train/test YOLO dataset builder; keep for historical retry3 reproduction
- legacy-copy: `/ghome/huangjd/code/detected/adpretrain_bridge/run_qm_xiepai_yolo26s_cls_stage.py` -> `/ghome/huangjd/code/detected/adpretrain_bridge/archive/20260518_engineering_cleanup/legacy_script_copies/run_qm_xiepai_yolo26s_cls_stage.py`; reason: legacy train/test YOLO runner; active val-threshold runner is run_qm_xiepai_yolo26s_cls_val_threshold_stage.py
- legacy-copy: `/ghome/huangjd/code/detected/adpretrain_bridge/summarize_fewshot_curve_train_test.py` -> `/ghome/huangjd/code/detected/adpretrain_bridge/archive/20260518_engineering_cleanup/legacy_script_copies/summarize_fewshot_curve_train_test.py`; reason: legacy train/test summarizer; active val-threshold summarizer is summarize_val_threshold_results.py

## Active Scripts

- `threshold_policies.py`
- `prepare_qm_xiepai_6_1_val_threshold_splits.py`
- `prepare_qm_xiepai_yolo26s_cls_val_threshold_stages.py`
- `run_fewshot_ahl_stage_val_threshold.py`
- `evaluate_fewshot_stage_metrics_val_threshold.py`
- `run_qm_xiepai_yolo26s_cls_val_threshold_stage.py`
- `summarize_val_threshold_results.py`
- `export_qm_xiepai_clip_features.py`
- `export_plain_features.py`
- `run_ahl_no_cudnn.py`

## Legacy Scripts Kept In Project Root

- `prepare_qm_xiepai_yolo26s_cls_stages.py`: legacy train/test YOLO dataset builder; keep for historical retry3 reproduction
- `run_qm_xiepai_yolo26s_cls_stage.py`: legacy train/test YOLO runner; active val-threshold runner is run_qm_xiepai_yolo26s_cls_val_threshold_stage.py
- `summarize_fewshot_curve_train_test.py`: legacy train/test summarizer; active val-threshold summarizer is summarize_val_threshold_results.py

## Active Output Roots

- `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_adpretrain_only_clip_b16_official_full_original_20260513`
- `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_adpretrain_only_clip_b16_official_train_test_20260512`
- `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_train_test_20260512`
- `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_plain_train_test_20260512`
- `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_full_original_clip_b16_plain_ahl_20260512`
- `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_6_1_val_threshold_20260517`
- `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_yolo26s_cls_train_test_20260513_retry3_recovered_eval`
