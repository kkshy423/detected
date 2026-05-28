# Safe Cleanup Manifest

Archive root: `/ghome/huangjd/code/detected/adpretrain_bridge/archive/20260515_safe_cleanup`

| Status | Path | Reason |
|---|---|---|
| archived | `check_fewshot_splits.py` | legacy calibration-split checker; current split uses check_qm_xiepai_fewshot_train_test_splits.py |
| archived | `prepare_qm_xiepai_fewshot_splits.py` | legacy calibration-split generator; current split uses prepare_qm_xiepai_fewshot_train_test_splits.py |
| archived | `summarize_fewshot_curve.py` | legacy calibration-split summarizer; current split uses summarize_fewshot_curve_train_test.py |
| archived | `run_fewshot_ahl_stage.py` | legacy calibration-split AHL runner; current split uses run_fewshot_ahl_stage_train_test.py |
| archived | `evaluate_fewshot_stage_metrics.py` | legacy calibration-split evaluator; current split uses evaluate_fewshot_stage_metrics_train_test.py |
| archived | `evaluate_qm_xiepai_adpretrain_only_train_test.py` | older non-official ADPretrain-only evaluator; official evaluator is active |
| archived | `recover_qm_xiepai_yolo26s_cls_retry3_eval.py` | one-off recovery script; recovered metrics already materialized, retry3 source weights are no longer present |
| archived | `cleanup_qm_xiepai_20260513.py` | one-off cleanup script; manifest already archived |
| archived | `yolo26n.pt` | unused YOLO nano checkpoint; current baseline is YOLO26s-cls |
| archived | `summary_baseline_xidun` | old baseline summary directory, not part of current qm_xiepai mainline |
| archived | `pbs/ahl_adp_plain_mqs.out` | old shared PBS stdout log; future PBS runs can recreate it |
| archived | `pbs/ahl_adp_plain_mqs.err` | old shared PBS stderr log; future PBS runs can recreate it |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S0_20260511.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S0_20260511_retry1.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S0_20260511_retry2.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S0_20260511_retry3.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S0_20260511_retry4.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S2_20260511.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S2_20260511_retry1.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S2_20260511_retry2.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S2_20260511_retry3.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S2_20260511_retry4.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S4_20260511.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S4_20260511_retry1.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S4_20260511_retry2.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S4_20260511_retry3.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S6_20260511.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S6_20260511_retry1.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S6_20260511_retry2.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S6_20260511_retry3.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S8_20260511.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S8_20260511_retry1.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S8_20260511_retry2.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_S8_20260511_retry3.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_chmm_eval_timing_S0_S4_20260511.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_export_chmm_20260511.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_export_chmm_20260511_retry1.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_export_chmm_20260511_retry2.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_fewshot_clip_b16_export_chmm_20260511_retry3.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_yolo26s_cls_train_test_S1_20260513_retry3.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_yolo26s_cls_train_test_S2_20260513_retry3.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_yolo26s_cls_train_test_S3_20260513_retry3.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_yolo26s_cls_train_test_S4_20260513_retry3.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_yolo26s_cls_train_test_S5_20260513_retry3.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_yolo26s_cls_train_test_S6_20260513_retry3.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_yolo26s_cls_train_test_S7_20260513_retry3.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_yolo26s_cls_train_test_S8_20260513_retry3.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_run/run_qm_xiepai_yolo26s_retry3_recovered_eval_20260513.sh` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S0_20260511.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S0_20260511_retry1.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S0_20260511_retry2.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S0_20260511_retry3.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S0_20260511_retry4.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S2_20260511.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S2_20260511_retry1.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S2_20260511_retry2.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S2_20260511_retry3.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S2_20260511_retry4.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S4_20260511.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S4_20260511_retry1.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S4_20260511_retry2.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S4_20260511_retry3.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S6_20260511.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S6_20260511_retry1.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S6_20260511_retry2.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S6_20260511_retry3.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S8_20260511.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S8_20260511_retry1.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S8_20260511_retry2.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_S8_20260511_retry3.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_chmm_eval_timing_S0_S4_20260511.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_export_chmm_20260511.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_export_chmm_20260511_retry1.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_export_chmm_20260511_retry2.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_fewshot_clip_b16_export_chmm_20260511_retry3.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_yolo26s_cls_train_test_S1_20260513_retry3.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_yolo26s_cls_train_test_S2_20260513_retry3.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_yolo26s_cls_train_test_S3_20260513_retry3.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_yolo26s_cls_train_test_S4_20260513_retry3.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_yolo26s_cls_train_test_S5_20260513_retry3.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_yolo26s_cls_train_test_S6_20260513_retry3.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_yolo26s_cls_train_test_S7_20260513_retry3.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_yolo26s_cls_train_test_S8_20260513_retry3.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| archived | `pbs/generated_submit/qm_xiepai_yolo26s_retry3_recovered_eval_20260513.pbs` | superseded or invalid generated PBS launcher/submit file; current outputs are already materialized |
| removed_cache | `__pycache__` | rebuildable Python bytecode cache |
| removed_empty_dir | `.locks` | empty stale lock directory |
| created | `qm_xiepai_current_results_20260515.md` | fresh active result index after cleanup |
