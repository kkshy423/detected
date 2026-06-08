# 20260607_cpu_gpu_uint8_full_stage_eval_v1 — Decision Handoff

## 一页结论

- 检测能力门槛: **8/12** cell PASS（F1 gap≤0.01 ∧ Recall gap≤0.02 ∧ AUC-PR gap≤0.005）。
- CPU 交叉校验漂移 cell: **6**（0 = 在线 ADP/AHL/bridge 复刻无偏）。
- 判定: **4 个 cell FAIL，需诊断**。

## 时间（median, ms, ref_and_query, AHL 主口径）

- preprocess: CPU 21.74 vs GPU 17.83
- AHL decoded-image-to-threshold: CPU 76.29 vs GPU 84.89
- 模型段地板 ahl_tensor_to_threshold: CPU 44.26 vs GPU 61.96（应近似相等）

## FAIL 阶段

- AHL-DINO S7 [fixed_cpu_threshold]: F1 gap 0.0282, Recall gap 0.0500, AUC-PR gap 0.0018
- AHL-DINO S7 [self_calibrated]: F1 gap 0.0220, Recall gap 0.0000, AUC-PR gap 0.0018
- ADP-AHL-bridge-v2 S7 [fixed_cpu_threshold]: F1 gap 0.0284, Recall gap 0.0500, AUC-PR gap 0.0000
- AHL-DINO S8 [self_calibrated]: F1 gap 0.0256, Recall gap 0.0500, AUC-PR gap 0.0000

## 推荐

见 master_performance_table / gap_summary / equivalence_diagnostics / time_table / cpu_cross_check。
F1 偶尔略超 CPU 一律按持平+噪声解读，不叙述 GPU 更优。
