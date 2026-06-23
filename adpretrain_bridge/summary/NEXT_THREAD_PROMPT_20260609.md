你接手工业异常检测项目 adpretrain_bridge 的实验执行线程。始终用中文回答；先看代码和已有结果再动手；优先 eval-only，不做低价值重复实验；GPU 任务必须走 PBS，绝不在登录节点直接跑 GPU；保持工作区干净，代码最小改动、可开关、可追溯、可回退、独立脚本优先；生产结论必须用 calib 选阈值、test 只用于最终报告（禁用 test oracle）；每次提交 PBS 都在 summary 下写提交清单。你是执行线程，每做完一项实验/修改要汇总一份精华交给理论/规划线程。

==================== 一、连接远程与环境 ====================

远程登录（Windows 本地，注意 key 路径加引号）：
  ssh -i "C:\Users\64541\.ssh\id_rsa_ustc" -p 39099 huangjd@202.38.69.241

主项目目录：/ghome/huangjd/code/detected/adpretrain_bridge
相关项目：/ghome/huangjd/code/detected/{ADPretrain,AHL}

conda python（run 脚本/登录节点直接用全路径，不要 conda activate）：
  /gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python
  版本：torch 2.4.1+cu121 / torchvision 0.19.1 / PIL 10.4.0 / numpy 1.24.4

ssh 偶发卡顿，命令加 -o ConnectTimeout=20，并用 timeout 包裹；登录节点 grep 掉 "WARNING|quantum|decrypt|upgraded|openssh" 噪声行。
读远程代码：scp 到本地工作目录再用编辑器读（Windows 上 /tmp 不可直接读）。

==================== 二、PBS 提交方式 ====================

模板：pbs/pbs_test.pbs，默认资源行 `#PBS -l nodes=1:gpus=1:a`（未明确授权不要改）。
PBS 不在模板里激活 conda；模板用 `startdocker -s <run.sh> bit:5000/deepo` 跑容器，conda 用 python 全路径写在 run 脚本里。
- run 脚本放 pbs/generated_run/，pbs 文件放 pbs/generated_submit/。
- 生成 PBS 只改 #PBS -N、-o、-e、SCRIPT_FILE_PATH。
- 提交：qsub pbs/generated_submit/xxx.pbs；查状态：qstat <jobid>（C=完成出队/R=运行/Q=排队）；日志在 pbs/<name>.out 和 .err。
run 脚本范式：
  #!/usr/bin/env bash
  set -euo pipefail
  cd /ghome/huangjd/code/detected/adpretrain_bridge
  exec /gdata1/huangjd/miniconda3/envs/adpretrain_ahl_bridge/bin/python <script.py> <args...>

==================== 三、数据 / split / 模型 ====================

主数据集：/gdata1/huangjd/xidun_mvtec_format_6_1/models__球面斜拍
有效 split：splits/20260529_qm_xiepai_6_1_fixed_180_70_val49（S8: train 420N+80D / calib 100N+49D / test 180N+70D；calib/test 跨 9 stage 同图）
backbone dinov2-large：权重 /gdata1/huangjd/model/dinov2-large/dinov2_vitl14_pretrain.pth；projector /ghome/huangjd/code/detected/ADPretrain/checkpoints/dino-large/checkpoints_img_norm.pth
alias = models_qiumianxiepai

==================== 四、本段已完成的工作（GPU uint8 预处理路线） ====================

目标：判断 GPU 预处理能否替代 CPU-PIL 以降低在线延迟。

根因（已确证）：旧 GPU 路径错在「先 div(255) 进 float 域再 bicubic resize」；PIL 是「uint8 域 resize + round/clip」。bicubic 过冲在 float 域被保留、uint8 域被量化抹掉 → feature 偏移 → 检测退化。
修复：新 backend `gpu_tensor_uint8_aa_true` —— GPU 上保持 uint8 域 resize/center_crop，crop 后才 cast→float/255→normalize（torchvision 0.19 CUDA uint8 bicubic+antialias 原生支持）。tensor 层 normalize 后 p95 diff=0（95% 像素 bit-exact）。

结论分层（务必照此判读，勿混读）：
1. ADP-only（training-free）：GPU 与 CPU 实质等价，全 9 stage 混淆矩阵几乎一致，唯一残差 AUC-PR≈0.0104（DINO encoder 对 ±1/255 边界像素浮点放大，物理下限非 bug）。
2. bridge v2：主线区 S1-S7 近持平（ΔF1≤0.027），S8 退化（ΔF1 0.069/ΔR 0.100）。
3. AHL 单方法：GPU 离线重训后波动过大、部分异常（S0 AUPR 0.213 崩溃、S8 ΔR 0.257），疑 few-shot 重训不稳定，**未归因、不可直接采信**。
速度：preprocess GPU≈16-19ms vs CPU≈32-42ms（粗粒度口径），真正 gpu_resize 仅 0.42ms；模型段 encoder≈44ms 是与 backend 无关的共同地板。
决策：GPU(gpu_tensor_uint8) 定为后续主指标，CPU 为对照。

==================== 五、关键脚本与作用 ====================

改过（已备份 .bak）：
- benchmark_preprocess_equivalence_paired.py：在线 ADP→AHL 配对 benchmark，5 backend（cpu_pil/旧float/aa_true/aa_false/gpu_tensor_uint8_aa_true），细粒度计时。
- benchmark_qm_xiepai_s8_adp_to_ahl_e2e.py：S8 端到端粗粒度计时（生产节拍基准口径），--preprocess-backend cpu_pil/gpu_tensor/gpu_tensor_uint8。
新建：
- benchmark_full_stage_cpu_gpu_eval.py：全阶段 S0-S8×两backend×两阈值在线对比，一次前向派生 ADP-only/AHL/bridge 三 score。
- export_gpu_uint8_plain_features.py：GPU uint8 预处理导出 ADP feature cache(.npy)，离线链路第一步。
- evaluate_adp_only_gpu_uint8.py：GPU 预处理的 ADP-only 离线 scores.csv（9 stage）。
- fuse_bridge_and_compare.py：读 ADP+AHL scores.csv → robust_norm+alpha 融合 bridge → stage_recall_bestf1 选阈值 → 出 GPU-vs-v2 对比表（纯 CPU，登录节点可跑）。
复用（未改本体）：run_fewshot_ahl_stage_val_threshold.py（单 stage AHL 训练+eval）、evaluate_fewshot_stage_metrics_val_threshold.py（result.txt→scores→metrics）、evaluate_qm_xiepai_adpretrain_only_fixed_180_79.py（ADP-only score 公式权威源）、threshold_policies.py、evaluate_stage_recall_bestf1_policy.py、evaluate_stage_aware_bridge_stability_check_v1.py（bridge robust_norm+alpha）、common.py（ADP 管线）。

==================== 六、口径与超参（务必沿用以保证可比） ====================

- 预处理 backend gpu_tensor_uint8_aa_true；ADP num_ref=8，AHL n_ref=5（**必须分离！**）；feature_levels=4。
- AHL 训练：epochs=30, steps=20, batch=48, seed=20260517（与 v2 同参）。
- 阈值 stage_recall_bestf1：S0-S2 best-F1@Recall≥0.90 / S3-S6 @Recall≥0.85 / S7-S8 纯 best-F1；calib 选、test 报。
- bridge v2 alpha：S0 N/A；S1-2=0.70；S3-4=0.60；S5-7=0.35；S8=0.70（S8 回 0.70 因主线切 YOLO、bridge 退 fallback，非突变）。
- 时间主口径 decoded_image_to_threshold_ms（image decode 只记录不计入）。粗粒度口径(e2e脚本)=生产基准；细粒度(equivalence脚本)逐段 cuda.synchronize 会放大 CPU 子操作测量值，仅用于定位 resize 差异。

==================== 七、关键数据/产物路径 ====================

- GPU feature cache(899图)：/gdata1/huangjd/data/xidun_qm_xiepai_6_1_adpretrain_dino_large_val_threshold_cache/plain_dino_large_norm_gpu_uint8_offline_v1；stage roots：.../stage_roots_gpu_uint8_offline_v1
- GPU 重训 AHL 输出：output/20260608_gpu_uint8_offline_full_retrain_v1/stages/S*/
- GPU ADP-only scores：output/20260608_gpu_uint8_offline_full_retrain_v1_adp_only/stages/S*/metrics/scores.csv
- v2 权威对照(CPU 离线)：summary/20260603_adp_ahl_stage_aware_bridge_v2_main_table/final_bridge_v2_main_table.csv
- 本段三实验 summary：summary/{20260607_gpu_preprocess_uint8_equiv_v1, 20260607_cpu_gpu_uint8_full_stage_eval_v1, 20260608_gpu_uint8_offline_full_retrain_v1}/
- 完整交接文档：summary/HANDOFF_gpu_uint8_worktrack_20260609.md

==================== 八、踩过的坑（避免重蹈） ====================

1. ADP/AHL n_ref 必须分离（ADP=8/AHL=5）；统一 8 → AHL forward 语义错、cross-check drift 16/26。
2. AHL 在线 vs 离线口径不同：差异在 feature/reference few-shot 采样，不在 score 后处理（离线 normalization 是恒等函数）；AHL 绝对值勿跨口径横读。
3. 计时两 backend 串行放同一 job 会顺序污染第二段 encoder/gpu_inference；要用独立 job。
4. scores.csv 两种 schema：ADP-only=`role,split,label,score,source_rel`；AHL 离线=`role,label,score,stage_rel,source_rel`（无 split，source_rel 带 split 前缀）；对齐用 role+basename，split 从 role 前缀推。
5. 对比表 md 里 bridge 行 alpha 列非空、ADP/AHL 行空，视觉上易与 precision 列错位；数值本身自洽（已用混淆矩阵验证，如 S6 bridge alpha=0.35 P=0.878 R=0.929 F1=0.903 TP65/FP9/TN171/FN5）。

==================== 九、PBS 历史（资源行均 #PBS -l nodes=1:gpus=1:a，A40） ====================

uint8 等价 210671/210672；全阶段在线 210717/211049(smoke)、210718(旧nRef bug已废)、211051(最终)；计时 211173(串行污染)、211179+211180(独立干净)；GPU 离线重训 211182(smoke)、211183(全量训练)、211185(ADP-only)。

==================== 十、GitHub ====================

仓库 git@github.com:kkshy423/detected.git（master 分支）。远程 ~/.ssh/id_rsa 已认证为 kkshy423；远程无 gh CLI。已推纯代码(219文件/3.56MB)，.gitignore 排除 *.pt/*.pkl/*.pth/*.npy + DRA/ + AITEX_anomaly_detection/ + */output/ + archive/；AHL 原残缺 submodule 已转普通目录(.git 备份在 /tmp/AHL_dotgit_backup_20260608)。最新 commit d11438e。
后续更新：cd /ghome/huangjd/code/detected && git add -A && git commit -m "..." && GIT_SSH_COMMAND='ssh -i ~/.ssh/id_rsa' git push

==================== 十一、头号待办 ====================

AHL 单方法 GPU-vs-v2 差异未归因。需补一个对照实验：用 CPU 预处理走完全相同的「重训一次」流程（同 seed 同超参），看 CPU 重训 vs v2 原表的波动有多大。
- 若 CPU 重训也大幅波动 → 波动来自 few-shot 训练随机性，AHL 单方法对比本身不稳，GPU 不背锅。
- 若 CPU 重训能复现 v2 → GPU 预处理确实显著伤 AHL。
此对照完成前，AHL 单方法的 GPU-vs-v2 结论悬置。ADP-only 等价 + bridge 主线区近持平已是可用结论。

先吸收以上信息（可 ssh 上去核对关键文件与 summary），确认你了解现状后告诉我"ok 了"，再开始新任务。
