# 20260527 S8 Online Time Profile PBS Submission

- job name: `s8_online_time`
- qsub id: `207940.Ghead`
- pbs path: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/s8_online_time_profile.pbs`
- run script: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_20260527_s8_online_time_profile.sh`
- dependency: none
- submit time: `2026-05-27 21:03:55 +08:00`
- resource line: `#PBS -l nodes=1:gpus=1:a`

## Output Roots

- AHL online timing output: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260527_s8_online_time_profile_ahl_adp`
- threshold compare output: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260527_s8_online_time_profile_threshold_compare`
- ADPretrain projected feature cache: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_clip_b16_val_threshold_cache/20260527_s8_online_time_profile_projected_adp`
- AHL stage root cache: `/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_clip_b16_val_threshold_cache/20260527_s8_online_time_profile_stage_roots`

## Notes

- Scope: latest fixed 180/79 split, `S8` only.
- ADPretrain timing now includes image load/transform, encoder, reference matching, residual, projector, and compression to AHL feature tensors.
- Feature file saving is recorded separately and excluded from online inference total.
- AHL eval-only reuses the existing S8 trained model and preloads reference feature batches before evaluation to approximate deployment.
- AHL training is skipped.
