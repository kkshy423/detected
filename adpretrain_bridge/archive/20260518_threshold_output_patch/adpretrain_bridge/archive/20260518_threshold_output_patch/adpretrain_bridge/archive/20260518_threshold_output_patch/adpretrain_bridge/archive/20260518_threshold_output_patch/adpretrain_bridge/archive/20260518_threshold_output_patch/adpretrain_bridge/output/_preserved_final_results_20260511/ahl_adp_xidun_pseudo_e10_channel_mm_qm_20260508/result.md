# Result Status

状态：已提交 PBS，等待/运行中。

## 作业信息

- Job ID：`203976.Ghead`
- PBS job name：`ahl_adp_chmm_qm`
- launcher：`/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_ahl_adp_xidun_pseudo_e10_channel_mm_qm_20260508.sh`
- PBS：`/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/ahl_adp_xidun_pseudo_e10_channel_mm_qm_20260508.pbs`
- cache：`/gdata1/huangjd/data/xidun_mvtec_format_adpretrain_xidun_pseudo_e10_channel_mm_qm_20260508`
- class alias：`models_qiumianxiepai_chmm`

## 已完成

- 已生成 per-channel moment matching 脚本
- 已生成 `.sh/.pbs`
- 已确认 `.sh` 无 BOM 且以 `#!/usr/bin/env bash` 开头
- 已确认 PBS 只修改 `#PBS -N` 和 `SCRIPT_FILE_PATH`
- 已确认 PBS 文件结尾两个空行
- 已按用户确认提交 PBS

## 待完成

- 生成 per-channel moment matching cache
- 运行缓存检查
- AHL 30 epoch
- 写出 aligned metrics
- 与 global scalar moment matching 对照