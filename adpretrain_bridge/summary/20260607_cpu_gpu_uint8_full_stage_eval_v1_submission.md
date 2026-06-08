# 提交清单 — 20260607_cpu_gpu_uint8_full_stage_eval_v1

收官对比实验：全阶段 S0-S8 × 两方法(ADP-only / AHL，bridge v2 派生) × 两 backend(cpu_pil / gpu_tensor_uint8_aa_true) × 两阈值模式(固定CPU阈值 / 各自标定)。
不重训、不调 alpha、不换 split、不加新 backend、不追 P95≤0.02、不做 async。

## 代码
- 新建独立脚本：`benchmark_full_stage_cpu_gpu_eval.py`（import 现有 benchmark + common，不改其行为；可回退）。
- 复用：ADP-only score 公式（evaluate_qm_xiepai_adpretrain_only_fixed_180_79.py）、bridge v2 alpha 四档（v2 main table 权威）、robust median/MAD 归一化、stage_recall_bestf1 阈值（evaluate_stage_recall_bestf1_policy.py）。
- 关键参数：NUM_REF=8（对齐 ADP-only 权威表）、FEATURE_LEVELS=4、tf32=False（两 backend 一致）、seed=0、warmup=5。
- encoder 每图每 backend 只前向一次，同一份 projected 派生 ADP-only/AHL(各阶段头)/bridge 三 score。calib/test 跨 stage 相同（从 S8 manifest 取），ref bank 每 backend 建一次全 stage 共用。

## PBS 提交记录

| job_id | 用途 | pbs 文件 | 资源行 | 状态 |
| --- | --- | --- | --- | --- |
| 210717.Ghead | smoke（S7,S8 + max-test 20 + warmup 2），验证 loop/bridge/阈值/计时/交叉校验结构 | pbs/generated_submit/pbs_20260607_cpu_gpu_uint8_full_stage_eval_v1_smoke.pbs | #PBS -l nodes=1:gpus=1:a | 成功(C) |
| 210718.Ghead | 全量 S0-S8 × 2 backend × 2 阈值模式 | pbs/generated_submit/pbs_20260607_cpu_gpu_uint8_full_stage_eval_v1.pbs | #PBS -l nodes=1:gpus=1:a | 运行中→待补状态 |

run 脚本：
- pbs/generated_run/run_20260607_cpu_gpu_uint8_full_stage_eval_v1_smoke.sh
- pbs/generated_run/run_20260607_cpu_gpu_uint8_full_stage_eval_v1.sh

均用 conda env adpretrain_ahl_bridge，节点 G141/A40。

## smoke 验证结论（210717）
- 流程、报表结构、计时拆分、bridge(S7/S8)、两阈值模式全部正确。
- ADP-only 在 smoke 子集即 F1/Recall gap=0（training-free，对量化鲁棒）。
- 子集 FAIL 全是 20 张统计噪声（pred_changed 0-1、Recall gap=0.05=1样本），非系统性退化；全量会收敛。
- 计时方向正确：preprocess median CPU 21.47ms vs GPU 5.12ms；cpu_resize/gpu_resize 互斥正确。

## 输出（待全量完成补全）
- output/20260607_cpu_gpu_uint8_full_stage_eval_v1/ 与 summary/ 同名目录：
  master_performance_table.{md,csv}, gap_summary.md, equivalence_diagnostics.csv,
  time_table.{md,csv}, cpu_cross_check.{md,csv}, decision_handoff.md, config_snapshot.json
- PBS .out：pbs/full_stage_eval_full.out
