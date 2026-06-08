# 20260607_cpu_gpu_uint8_full_stage_eval_v1 — Decision Handoff

## 一页结论

- 检测能力门槛: **1/52** cell PASS（F1 gap≤0.01 ∧ Recall gap≤0.02 ∧ AUC-PR gap≤0.005）。
- CPU 交叉校验漂移 cell: **16**（0 = 在线 ADP/AHL/bridge 复刻无偏）。
- 判定: **51 个 cell FAIL，需诊断**。

## 时间（median, ms, ref_and_query, AHL 主口径）

- preprocess: CPU 42.08 vs GPU 20.02
- AHL decoded-image-to-threshold: CPU 81.68 vs GPU 73.67
- 模型段地板 ahl_tensor_to_threshold: CPU 34.03 vs GPU 34.16（应近似相等）

## FAIL 阶段

- ADP-only-DINO S0 [fixed_cpu_threshold]: F1 gap 0.0000, Recall gap 0.0000, AUC-PR gap 0.0104
- ADP-only-DINO S0 [self_calibrated]: F1 gap 0.0170, Recall gap 0.0000, AUC-PR gap 0.0104
- AHL-DINO S0 [self_calibrated]: F1 gap 0.0163, Recall gap 0.0000, AUC-PR gap 0.0017
- ADP-only-DINO S1 [fixed_cpu_threshold]: F1 gap 0.0000, Recall gap 0.0000, AUC-PR gap 0.0104
- ADP-only-DINO S1 [self_calibrated]: F1 gap 0.0170, Recall gap 0.0000, AUC-PR gap 0.0104
- AHL-DINO S1 [fixed_cpu_threshold]: F1 gap 0.0093, Recall gap 0.0000, AUC-PR gap 0.0087
- AHL-DINO S1 [self_calibrated]: F1 gap 0.0341, Recall gap 0.0000, AUC-PR gap 0.0087
- ADP-AHL-bridge-v2 S1 [fixed_cpu_threshold]: F1 gap 0.0016, Recall gap 0.0143, AUC-PR gap 0.0095
- ADP-AHL-bridge-v2 S1 [self_calibrated]: F1 gap 0.0162, Recall gap 0.0000, AUC-PR gap 0.0095
- ADP-only-DINO S2 [fixed_cpu_threshold]: F1 gap 0.0000, Recall gap 0.0000, AUC-PR gap 0.0104
- ADP-only-DINO S2 [self_calibrated]: F1 gap 0.0170, Recall gap 0.0000, AUC-PR gap 0.0104
- AHL-DINO S2 [fixed_cpu_threshold]: F1 gap 0.0017, Recall gap 0.0000, AUC-PR gap 0.0118
- AHL-DINO S2 [self_calibrated]: F1 gap 0.0151, Recall gap 0.0000, AUC-PR gap 0.0118
- ADP-AHL-bridge-v2 S2 [fixed_cpu_threshold]: F1 gap 0.0011, Recall gap 0.0143, AUC-PR gap 0.0093
- ADP-AHL-bridge-v2 S2 [self_calibrated]: F1 gap 0.0255, Recall gap 0.0143, AUC-PR gap 0.0093
- ADP-only-DINO S3 [fixed_cpu_threshold]: F1 gap 0.0046, Recall gap 0.0000, AUC-PR gap 0.0104
- ADP-only-DINO S3 [self_calibrated]: F1 gap 0.0392, Recall gap 0.0286, AUC-PR gap 0.0104
- AHL-DINO S3 [fixed_cpu_threshold]: F1 gap 0.0163, Recall gap 0.0000, AUC-PR gap 0.0075
- AHL-DINO S3 [self_calibrated]: F1 gap 0.0064, Recall gap 0.0000, AUC-PR gap 0.0075
- ADP-AHL-bridge-v2 S3 [fixed_cpu_threshold]: F1 gap 0.0072, Recall gap 0.0143, AUC-PR gap 0.0084
- ADP-AHL-bridge-v2 S3 [self_calibrated]: F1 gap 0.0159, Recall gap 0.0143, AUC-PR gap 0.0084
- ADP-only-DINO S4 [fixed_cpu_threshold]: F1 gap 0.0046, Recall gap 0.0000, AUC-PR gap 0.0104
- ADP-only-DINO S4 [self_calibrated]: F1 gap 0.0392, Recall gap 0.0286, AUC-PR gap 0.0104
- AHL-DINO S4 [fixed_cpu_threshold]: F1 gap 0.0234, Recall gap 0.0000, AUC-PR gap 0.0102
- AHL-DINO S4 [self_calibrated]: F1 gap 0.0668, Recall gap 0.0000, AUC-PR gap 0.0102
- ADP-AHL-bridge-v2 S4 [fixed_cpu_threshold]: F1 gap 0.0099, Recall gap 0.0000, AUC-PR gap 0.0071
- ADP-AHL-bridge-v2 S4 [self_calibrated]: F1 gap 0.0283, Recall gap 0.0000, AUC-PR gap 0.0071
- ADP-only-DINO S5 [fixed_cpu_threshold]: F1 gap 0.0046, Recall gap 0.0000, AUC-PR gap 0.0104
- ADP-only-DINO S5 [self_calibrated]: F1 gap 0.0392, Recall gap 0.0286, AUC-PR gap 0.0104
- AHL-DINO S5 [fixed_cpu_threshold]: F1 gap 0.0030, Recall gap 0.0000, AUC-PR gap 0.0062
- AHL-DINO S5 [self_calibrated]: F1 gap 0.0516, Recall gap 0.0000, AUC-PR gap 0.0062
- ADP-AHL-bridge-v2 S5 [fixed_cpu_threshold]: F1 gap 0.0025, Recall gap 0.0143, AUC-PR gap 0.0056
- ADP-AHL-bridge-v2 S5 [self_calibrated]: F1 gap 0.0543, Recall gap 0.0143, AUC-PR gap 0.0056
- ADP-only-DINO S6 [fixed_cpu_threshold]: F1 gap 0.0046, Recall gap 0.0000, AUC-PR gap 0.0104
- ADP-only-DINO S6 [self_calibrated]: F1 gap 0.0392, Recall gap 0.0286, AUC-PR gap 0.0104
- AHL-DINO S6 [fixed_cpu_threshold]: F1 gap 0.0184, Recall gap 0.0286, AUC-PR gap 0.0076
- AHL-DINO S6 [self_calibrated]: F1 gap 0.0111, Recall gap 0.0429, AUC-PR gap 0.0076
- ADP-AHL-bridge-v2 S6 [fixed_cpu_threshold]: F1 gap 0.0028, Recall gap 0.0143, AUC-PR gap 0.0087
- ADP-AHL-bridge-v2 S6 [self_calibrated]: F1 gap 0.0097, Recall gap 0.0000, AUC-PR gap 0.0087
- ADP-only-DINO S7 [fixed_cpu_threshold]: F1 gap 0.0000, Recall gap 0.0000, AUC-PR gap 0.0104
- ADP-only-DINO S7 [self_calibrated]: F1 gap 0.0023, Recall gap 0.0143, AUC-PR gap 0.0104
- AHL-DINO S7 [fixed_cpu_threshold]: F1 gap 0.0171, Recall gap 0.0286, AUC-PR gap 0.0086
- AHL-DINO S7 [self_calibrated]: F1 gap 0.0259, Recall gap 0.0429, AUC-PR gap 0.0086
- ADP-AHL-bridge-v2 S7 [fixed_cpu_threshold]: F1 gap 0.0326, Recall gap 0.0429, AUC-PR gap 0.0091
- ADP-AHL-bridge-v2 S7 [self_calibrated]: F1 gap 0.0326, Recall gap 0.0429, AUC-PR gap 0.0091
- ADP-only-DINO S8 [fixed_cpu_threshold]: F1 gap 0.0000, Recall gap 0.0000, AUC-PR gap 0.0104
- ADP-only-DINO S8 [self_calibrated]: F1 gap 0.0023, Recall gap 0.0143, AUC-PR gap 0.0104
- AHL-DINO S8 [fixed_cpu_threshold]: F1 gap 0.0165, Recall gap 0.0286, AUC-PR gap 0.0119
- AHL-DINO S8 [self_calibrated]: F1 gap 0.0223, Recall gap 0.0286, AUC-PR gap 0.0119
- ADP-AHL-bridge-v2 S8 [fixed_cpu_threshold]: F1 gap 0.0613, Recall gap 0.1000, AUC-PR gap 0.0097
- ADP-AHL-bridge-v2 S8 [self_calibrated]: F1 gap 0.1093, Recall gap 0.1714, AUC-PR gap 0.0097

## 推荐

见 master_performance_table / gap_summary / equivalence_diagnostics / time_table / cpu_cross_check。
F1 偶尔略超 CPU 一律按持平+噪声解读，不叙述 GPU 更优。
