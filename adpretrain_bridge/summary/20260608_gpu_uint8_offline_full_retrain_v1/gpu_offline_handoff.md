# GPU 离线重训 vs v2(CPU) 对比 — 结果与重要 caveat（交理论线程）

实验名: `20260608_gpu_uint8_offline_full_retrain_v1`
日期: 2026-06-08
链路: GPU uint8 预处理 → GPU feature cache(899图) → 9 stage 重训 AHL(epochs=30,n_ref=5,seed=20260517,与v2同参) → 离线 test → ADP-only/AHL/bridge scores → stage_recall_bestf1 选阈值 → 与 v2 main table 并读。
PBS: 211182(smoke,C) / 211183(全量训练,C) / 211185(ADP-only,C)。

## 核心结论（分三层，诚实判读）

### 1. ADP-only：稳定，符合预期
全 9 stage ΔF1 0.002–0.039，AUPR 恒 0.9047(GPU) vs 0.9151(v2 CPU)，差 0.0104。
training-free，差异纯为 GPU/CPU resize 的浮点残差。**可信、实质等价。**

### 2. bridge v2：主线区稳，S8 退化
- S1-S7 ΔF1 ≤0.027、ΔR ≤0.029，近持平（bridge 由 ADP 主导，AHL 波动被 alpha 稀释）。
- S8 ΔF1 0.069 / ΔR 0.100（GPU 漏检多）。S8 alpha=0.70 把 70% 权重给 ADP，但 AHL 那 30% 在 GPU 下退化明显，拖累 bridge。

### 3. AHL 单方法：波动过大，部分异常 —— **不可直接采信**
- S0 AUPR 0.213（接近随机，明显崩溃）、ΔF1 0.117；S5 ΔF1 0.159；S8 ΔR 0.257。
- 这种量级远超 ±1/255 预处理差异所能解释，即便经训练放大。
- **判断：AHL few-shot 重训（episode 采样+小样本+30epoch）本身高度不稳定，单次重训不可靠**；GPU 预处理的微小 feature 差异被训练不稳定性放大成大波动。与历史记录"GPU retrain 反而更差"一致。

## 关键 caveat（必须随表传递）

**本表的 AHL 单列不能解读为"GPU vs CPU 预处理的纯效应"**，因为它混入了 few-shot 重训的随机不稳定性。要严格归因，需要一个**对照实验**：用 **CPU 预处理走完全相同的"重训一次"流程**，看 CPU 重训 vs v2 原表的波动有多大。
- 若 CPU 重训也波动这么大 → 波动来自训练随机性，GPU 不背锅，AHL 单方法对比本身就不稳。
- 若 CPU 重训能复现 v2 → 则 GPU 预处理确实显著伤害 AHL。
该对照未做。**在它完成前，AHL 单方法的 GPU-vs-v2 差异不下结论。**

## 可采信的部分
- ADP-only GPU 等价（仅 AUPR 0.0104 resize 残差）。
- bridge 主线区(S1-S7) GPU 近持平；S8 退化。
- 这两条与上一轮在线全阶段(211051)结论方向一致，相互印证。

## 文件
- 对比表: `summary/20260608_gpu_uint8_offline_full_retrain_v1/gpu_offline_master_table.{md,csv}`
- GPU scores: `output/20260608_gpu_uint8_offline_full_retrain_v1/stages/S*/metrics/scores.csv`(AHL)、`..._adp_only/stages/S*/metrics/scores.csv`(ADP)
- v2 对照: `summary/20260603_adp_ahl_stage_aware_bridge_v2_main_table/final_bridge_v2_main_table.csv`
