# 20260608 encoder latency breakdown and S8 alpha probe v1

## Boundary

This is a diagnostic experiment bundle. It does not change algorithms, split, backbone, production alpha, or the bridge v2 main table.

Experiment A diagnoses encoder latency sources:

- TF32 OFF vs ON.
- synced per-forward timing vs pure total-forward synchronized timing.
- backbone metadata is recorded to check whether ViT-L/14 at 224 and batch=1 explains the tens-of-ms floor.

Experiment B probes S8 alpha direction only:

- S8 alpha in {0.70, 0.50, 0.35}.
- CPU/PIL and GPU `gpu_tensor_uint8_aa_true` both evaluated.
- Thresholds are selected on calib and test is only reported.
- File names and report text must state `diagnostic, not main-table update`.

## Hard rule for experiment B

Even if alpha=0.35 looks better, do not update the bridge v2 main table from this experiment. The online AHL/bridge score route and old v2 main table route are known to differ (`combine_dra_outputs` vs offline AHL test dataloader). Main-table alpha changes require AHL score口径对齐 first, which is not part of this run.
