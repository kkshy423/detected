# Invalid ADPretrain-only Cache-Norm Evaluation

This output was a temporary cache-norm sanity check and is not the official ADPretrain-only baseline.

Do not use this directory as the P0 baseline. Use instead:

`/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_adpretrain_only_clip_b16_official_train_test_20260512`

Reason: the cache-norm script scored the compressed AHL feature cache, not the official ADPretrain residual/projector/image-score pipeline.