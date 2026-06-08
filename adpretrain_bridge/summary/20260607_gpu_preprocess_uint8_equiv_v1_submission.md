# 提交清单 — 20260607_gpu_preprocess_uint8_equiv_v1

实验目标：在 GPU 上复刻 CPU-PIL 的预处理数值语义，验证能否恢复检测指标。
新增 backend：`gpu_tensor_uint8_aa_true`（在 GPU uint8 域做 bicubic+antialias resize/crop，resize 后再 cast→float/255→normalize；对齐 PIL 的 "uint8 域插值 + round/clip" 语义）。

## 代码改动
- 文件：`benchmark_preprocess_equivalence_paired.py`
  - 备份：`benchmark_preprocess_equivalence_paired.py.bak_20260607`
  - 新增函数 `prepare_gpu_uint8_from_pil`（uint8 域 resize/crop，cast 后置）
  - 注册到 `BACKENDS` 与 `prepare_backend_from_pil`
  - config_snapshot backends 描述补充一条
  - 最小改动、可回退（删 backend 名即停用，恢复 .bak 即还原）

## 诊断脚本（一次性，不属主代码）
- CPU 端 resize 等价性诊断：`/tmp/diag_resize_equiv.py`（登录节点跑，纯 CPU）
- GPU 自检：`output/20260607_gpu_preprocess_uint8_equiv_v1/gpu_selfcheck_uint8_resize.py`

## PBS 提交记录

| job_id | 用途 | pbs 文件 | 资源行 | 状态 |
| --- | --- | --- | --- | --- |
| 210671.Ghead | GPU self-check：CUDA uint8 bicubic 是否与 PIL/CPU 一致 | pbs/generated_submit/pbs_20260607_gpu_preprocess_uint8_equiv_v1_selfcheck.pbs | #PBS -l nodes=1:gpus=1:a | 成功(C) |
| 210672.Ghead | 完整 S8 等价性 eval（2 scenario × 5 backend） | pbs/generated_submit/pbs_20260607_gpu_preprocess_uint8_equiv_v1.pbs | #PBS -l nodes=1:gpus=1:a | 成功(C) |

run 脚本：
- pbs/generated_run/run_20260607_gpu_preprocess_uint8_equiv_v1_selfcheck.sh
- pbs/generated_run/run_20260607_gpu_preprocess_uint8_equiv_v1.sh

均运行于节点 G141 / A40，使用 conda env `adpretrain_ahl_bridge`。

## 输出
- output/20260607_gpu_preprocess_uint8_equiv_v1/{equivalence_fix_summary.md, equivalence_fix_metrics.csv, equivalence_fix_latency.csv, changed_samples_by_backend.csv, config_snapshot.json}
- summary/20260607_gpu_preprocess_uint8_equiv_v1/（同上副本）
- self-check stdout: pbs/uint8_equiv_selfcheck.out

## 核心结论（详见 equivalence_fix_summary.md）
- self-check：CUDA uint8 bicubic 与 CPU uint8 近 bit-exact（max 0.0073，p95=0）；GPU uint8 vs PIL 的 normalize-tensor p95=0。
- 新 backend 在所有维度均为最优 GPU 路径，但仍未通过严格接受标准。
- 详细判定与建议在精华报告 `theory_handoff_uint8_equiv.md`。
