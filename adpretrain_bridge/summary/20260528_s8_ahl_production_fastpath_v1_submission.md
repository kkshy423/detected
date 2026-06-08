# 20260528_s8_ahl_production_fastpath_v1 提交清单

- job_name: `ahl_s8_fast_v1`
- launcher: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_20260528_s8_ahl_production_fastpath_v1.sh`
- pbs: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/20260528_s8_ahl_production_fastpath_v1.pbs`
- output_root: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260528_s8_ahl_production_fastpath_v1`
- ahl_dir: `/ghome/huangjd/code/detected/adpretrain_bridge/output/20260528_s8_ahl_production_fastpath_v1/stages/S8/ahl`
- mode: S8 eval-only，复用已有 S8 AHL 权重，启用 `production_like_fast_path`
- timing_scope: 计时区排除 dataloader yield 等待、loss、指标计算、result 写盘、tqdm；保留输入上设备、reference attach、AHL forward、score postprocess
- result_dump: 保留 `result.txt` 便于一致性检查，但写盘在计时区外
- resource_policy: 从 `pbs/pbs_test.pbs` 复制；本次按用户明确要求将资源行改为 `#PBS -l nodes=1:gpus=2:b`，同时改 `#PBS -N` 和 `SCRIPT_FILE_PATH`
- resource_note: 当前 AHL 代码未启用 DataParallel/DDP/torchrun，虽然 PBS 分配 2 张 GPU，实际推理默认仍由单进程使用第一张可见 GPU。

## Command

```bash
/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python /ghome/huangjd/code/detected/adpretrain_bridge/_archive/cleanup_20260525/root_legacy_scripts/run_ahl_no_cudnn.py --dataset mvtecad --dataset_root /gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_clip_b16_val_threshold_cache/20260527_s8_time_profile_stage_roots_plain_fixed_180_79/S8 --experiment_dir /ghome/huangjd/code/detected/adpretrain_bridge/output/20260528_s8_ahl_production_fastpath_v1/stages/S8/ahl --classname models_qiumianxiepai --feat_classname models_qiumianxiepai --nAnomaly 80 --test_threshold 80 --know_class train_defect --cluster_num 2 --nRef 5 --epochs 30 --steps_per_epoch 20 --batch_size 48 --workers 4 --backbone resnet18 --ramdn_seed 20260517 --eval_only --load_weights /ghome/huangjd/code/detected/adpretrain_bridge/output/20260519_ahl_plain_fixed_180_79_stage_v3/stages/S8/ahl/models_qiumianxiepai_ctest.pkl --production_like_fast_path --keep_fast_result_dump
```
## Result Snapshot

- job_status: C exit_status=0
- output_summary: /ghome/huangjd/code/detected/adpretrain_bridge/output/20260528_s8_ahl_production_fastpath_v1/summary/s8_ahl_fastpath_result_summary.md
- score_consistency: max_abs_diff vs 20260527 online result 6.34e-08; AUC-ROC/AUC-PR 0.9148 / 0.8613
- latency_all: mean 8.45 ms, median 2.90 ms, P90 4.53 ms, P95 4.59 ms, max 2155.22 ms
- latency_steady_drop_first: mean 3.06 ms, median 2.90 ms, P90 4.52 ms, P95 4.59 ms, max 10.04 ms
- conclusion: production-like fast path removes the previous periodic slow-batch tail; the remaining max is the first-image CUDA/kernel cold-start spike, not steady online latency.