# 20260527 S8 Time Profile PBS Submission

- job name: `s8_time_profile`
- qsub id: `207904.Ghead`
- pbs path: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/s8_time_profile.pbs`
- run script: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_20260527_s8_time_profile.sh`
- dependency: none
- submit time: `2026-05-27 19:33:10 +08:00`
- resource line: `#PBS -l nodes=1:gpus=1:a`

## Output Roots

- AHL full flow: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260527_s8_time_profile_ahl_adp_fullflow`
- YOLO full supervised: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260527_s8_time_profile_yolo26s_fullsupervised`
- threshold compare: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260527_s8_time_profile_threshold_compare`
- ADPretrain feature cache: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_clip_b16_val_threshold_cache/20260527_s8_time_profile_plain_official_img_angle`
- AHL stage root cache: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_clip_b16_val_threshold_cache/20260527_s8_time_profile_stage_roots_plain_fixed_180_79`

## Notes

- Scope: latest fixed 180/79 split, `S8` only.
- ADPretrain: export official CLIP-B16 projected residual features for train/val/test.
- AHL: reuse existing trained S8 model and run eval-only inference/evaluation.
- YOLO: retrain YOLO26s-cls S8 full-supervised model and run inference/evaluation.
- Timing purpose: collect per-image ADPretrain feature time, AHL process time, full ADPretrain+AHL flow time, YOLO train time, and YOLO inference time.
