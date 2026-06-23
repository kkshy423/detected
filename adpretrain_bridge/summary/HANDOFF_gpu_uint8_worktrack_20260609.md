# GPU uint8 预处理路线 — 完整工作交接（给下一实验线程）

时间跨度: 2026-06-07 ~ 06-09
执行线程: 实验执行线程
项目: adpretrain_bridge 工业异常检测（球面斜拍 qm_xiepai 6_1）

## 0. 一页背景

目标: 判断「GPU 预处理能否替代 CPU-PIL」以降低在线延迟。历史上旧 GPU 路径(float 域 resize)
被否决（检测指标退化、重训也救不回）。本段工作找到根因并修复，做了全阶段对比、干净计时、
GPU 离线重训，并把代码推上 GitHub。

## 1. 核心技术结论（最重要）

**根因**: 旧 GPU 路径错在「先 div(255) 进 float 域再 resize」；PIL 是「uint8 域 resize + round/clip」。
bicubic 过冲在 float 域被保留、在 uint8 域被量化抹掉 → feature 偏移 → 检测退化。

**修复**: 新 backend `gpu_tensor_uint8_aa_true` —— GPU 上保持 **uint8 域 resize/crop，crop 后才 cast→float/255→normalize**。
torchvision 0.19 对 CUDA uint8 bicubic+antialias 原生支持。

**等价性验证（已确证）**:
- tensor 层面: GPU uint8 vs CPU-PIL，normalize 后 p95 diff = 0（95% 像素 bit-exact），残差仅极少数 ±1/255 量化边界像素。
- ADP-only（training-free）: 全 9 stage CPU/GPU 混淆矩阵几乎完全一致，唯一残差 AUC-PR 0.0104（DINO encoder 对边界像素的浮点放大，物理下限，非 bug）。
- 速度: preprocess GPU≈16-19ms vs CPU≈32-42ms（粗粒度口径）；真正 gpu_resize 仅 0.42ms；模型段(encoder~44ms)是共同地板，与预处理 backend 无关。

**当前结论分层**:
1. ADP-only: GPU 与 CPU 实质等价（仅 resize 浮点残差）。
2. bridge v2 主线区(S1-S7): GPU 近持平（ΔF1≤0.027）；S8 退化（ΔF1 0.069/ΔR 0.100）。
3. AHL 单方法: 重训波动过大、部分异常（S0 AUPR 0.213 崩溃、S8 ΔR 0.257），**不可直接采信**——疑似 few-shot 重训不稳定，需 CPU 重训对照才能归因（未做）。

**决策**: GPU(gpu_tensor_uint8) 定为后续主指标，CPU 为对照。

## 2. 关键脚本与作用（远程 /ghome/huangjd/code/detected/adpretrain_bridge/）

| 脚本 | 作用 | 状态 |
|---|---|---|
| `benchmark_preprocess_equivalence_paired.py` | 在线 ADP→AHL 配对 benchmark；含 5 个 backend（cpu_pil / 旧float / aa_true / aa_false / **gpu_tensor_uint8_aa_true**）；细粒度计时 | 已加 uint8 backend，备份 `.bak_20260607` |
| `benchmark_qm_xiepai_s8_adp_to_ahl_e2e.py` | S8 端到端**粗粒度计时**脚本（生产节拍基准口径）；含 `--preprocess-backend cpu_pil/gpu_tensor/gpu_tensor_uint8` | 已加 gpu_tensor_uint8，备份 `.bak_20260608` |
| `benchmark_full_stage_cpu_gpu_eval.py` | 全阶段 S0-S8 × 两 backend × 两阈值模式**在线**对比；一次前向派生 ADP-only/AHL/bridge 三 score | 新建 |
| `export_gpu_uint8_plain_features.py` | 用 GPU uint8 预处理导出 ADP feature cache(.npy)；离线链路第一步 | 新建 |
| `evaluate_adp_only_gpu_uint8.py` | GPU 预处理的 ADP-only 离线 scores.csv（9 stage） | 新建 |
| `fuse_bridge_and_compare.py` | 读 ADP+AHL scores.csv → robust_norm + alpha 融合 bridge → stage_recall_bestf1 选阈值 → 出 GPU-vs-v2 对比表（纯 CPU，登录节点可跑） | 新建 |

## 3. 复用的既有脚本（未改本体）

- `run_fewshot_ahl_stage_val_threshold.py` — 单 stage AHL 训练+eval（离线，调 AHL/main.py 出 result.txt）。参数 `--cache-root/--stage-root-base/--stage/--epochs/--n-ref`。
- `evaluate_fewshot_stage_metrics_val_threshold.py` — result.txt → scores.csv → metrics。
- `evaluate_qm_xiepai_adpretrain_only_fixed_180_79.py` — ADP-only score 公式权威源（`sqrt(‖patch‖+1)-1`→上采样+gaussian σ4→空间max；feature_levels=4, num_ref=8）。
- `threshold_policies.py` / `evaluate_stage_recall_bestf1_policy.py` — 阈值策略。
- `evaluate_stage_aware_bridge_stability_check_v1.py` — bridge robust_norm + alpha 权威实现。
- `common.py` — ADP 管线（encode_multiscale/match/residual/compress_four_to_two）。

## 4. 实验与产物（summary/ 下）

| 实验 | 目录 | 内容 |
|---|---|---|
| uint8 等价性修复 | `20260607_gpu_preprocess_uint8_equiv_v1/` | 根因确证 + S8 端到端；`theory_handoff_uint8_equiv.md` |
| 全阶段在线对比 | `20260607_cpu_gpu_uint8_full_stage_eval_v1/` | S0-S8×2backend×2阈值；master_performance/gap_summary/equivalence_diagnostics/cpu_cross_check/decision_handoff |
| GPU 离线重训 | `20260608_gpu_uint8_offline_full_retrain_v1/` | GPU 重训 AHL 后与 v2 并读；`gpu_offline_master_table.{md,csv}` + `gpu_offline_handoff.md`(含 caveat) |

## 5. 关键数据/路径

- GPU feature cache(899图): `.../plain_dino_large_norm_gpu_uint8_offline_v1`；stage roots `.../stage_roots_gpu_uint8_offline_v1`
- GPU 重训 AHL 输出: `output/20260608_gpu_uint8_offline_full_retrain_v1/stages/S*/`
- GPU ADP-only scores: `output/20260608_gpu_uint8_offline_full_retrain_v1_adp_only/stages/S*/metrics/scores.csv`
- v2 权威对照(CPU): `summary/20260603_adp_ahl_stage_aware_bridge_v2_main_table/final_bridge_v2_main_table.csv`
- split: `splits/20260529_qm_xiepai_6_1_fixed_180_70_val49`（calib 100N+49D / test 180N+70D）

## 6. 口径与超参（务必沿用以保证可比）

- backbone dinov2-large；alias models_qiumianxiepai；预处理 `gpu_tensor_uint8_aa_true`。
- **ADP num_ref=8；AHL n_ref=5（必须分离！）**；feature_levels=4。
- AHL 训练: epochs=30, steps=20, batch=48, seed=20260517（与 v2 同参）。
- 阈值 stage_recall_bestf1: S0-2 best-F1@R≥0.90 / S3-6 @R≥0.85 / S7-8 纯 best-F1；calib 选、test 报。
- bridge v2 alpha: S0 N/A；S1-2=0.70；S3-4=0.60；S5-7=0.35；S8=0.70（S8 回 0.70 因主线切 YOLO、bridge 退 fallback）。
- 时间主口径 decoded_image_to_threshold_ms（decode 只记录）。**粗粒度(e2e脚本)=生产基准；细粒度(equivalence脚本)逐段 synchronize 放大 CPU 测量值，仅定位 resize 差异用。**

## 7. 踩过的坑

1. **ADP/AHL n_ref 必须分离**（ADP=8/AHL=5）；统一 8 → AHL forward 语义错、cross-check drift 16/26。
2. **AHL 在线 vs 离线口径不同**: 差异在 feature/reference 采样，不在 score 后处理（离线 normalization 是恒等）。AHL 绝对值勿跨口径横读。
3. **计时两 backend 串行同 job 顺序污染**第二段 encoder；用独立 job。
4. **scores.csv 两种 schema**: ADP-only=`role,split,label,score,source_rel`；AHL=`role,label,score,stage_rel,source_rel`（无split，source_rel带split前缀）。对齐用 role+basename。
5. **md 表 alpha 列易误读**: bridge 行 alpha 非空、易与 precision 列视觉错位；数值已用混淆矩阵验证自洽（如 S6 bridge: alpha=0.35, P=0.878, R=0.929, F1=0.903, TP65/FP9/TN171/FN5）。

## 8. PBS 记录（资源行均 #PBS -l nodes=1:gpus=1:a，A40）

- uint8 等价: 210671/210672
- 全阶段在线: 210717/211049(smoke) / 210718(旧nRef bug废) / 211051(最终)
- 计时: 211173(串行污染) / 211179+211180(独立干净)
- GPU 离线重训: 211182(smoke) / 211183(全量训练) / 211185(ADP-only)

## 9. GitHub

仓库 `git@github.com:kkshy423/detected.git`(master)。远程 ~/.ssh/id_rsa 已认证 kkshy423；无 gh CLI。纯代码(219文件/3.56MB)，.gitignore 排除 *.pt/pkl/pth/npy+DRA/+AITEX/+*/output/+archive/。AHL 转普通目录(.git 备份 /tmp/AHL_dotgit_backup_20260608)。最新 commit d11438e。更新: `cd /ghome/huangjd/code/detected && git add -A && git commit && GIT_SSH_COMMAND='ssh -i ~/.ssh/id_rsa' git push`。

## 10. 未决 / 建议下一步

- **AHL 单方法 GPU-vs-v2 差异未归因**: 需补「CPU 预处理走相同重训流程」对照，判定波动是 few-shot 训练随机性还是 GPU 预处理。此前 AHL 单列不下结论。
- ADP-only 等价 + bridge 主线区近持平已可用。
- 继续加速且不改数值语义: CPU-PIL+预加载/服务化；或确认产线交付已解码 array（GPU 路径可省 pil_to_tensor ~5-17ms）。

## 11. 工作风格约束（继承）

中文回答；先看代码再动手；eval-only 优先；GPU 必走 PBS（不在登录节点跑 GPU）；最小改动/可开关/可回退/独立脚本优先；calib 选阈值 test 只报告；提交 PBS 写清单。
python: `/gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python`（torch 2.4.1+cu121 / tv 0.19.1 / PIL 10.4.0）。
远程: `ssh -i C:\Users\64541\.ssh\id_rsa_ustc -p 39099 huangjd@202.38.69.241`；主项目 `/ghome/huangjd/code/detected/adpretrain_bridge`。
