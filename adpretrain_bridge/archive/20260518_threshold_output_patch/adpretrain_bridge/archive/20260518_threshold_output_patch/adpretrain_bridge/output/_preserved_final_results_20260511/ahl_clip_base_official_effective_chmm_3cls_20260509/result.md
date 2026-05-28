# ahl_clip_base_official_effective_chmm_3cls_20260509

Status: submitted PBS, waiting for completion.

- Job ID: `204118.Ghead`
- PBS job name: `clip_off_3cls`
- dataset root: `/gdata1/huangjd/data/xidun_mvtec_format_threecls`
- alias profile: `xidun3`
- backbone: `clip-base`
- projector: `checkpoints_img_angle.pth` as CLIP official effective projector
- plain cache: `/gdata1/huangjd/data/xidun_mvtec_format_threecls_adpretrain_clip_base_official_effective_20260509`
- CHMM cache: `/gdata1/huangjd/data/xidun_mvtec_format_threecls_adpretrain_clip_base_official_effective_chmm_20260509`
- AHL cluster_num: `2`
- epochs: `30`
- metrics target: `/ghome/huangjd/code/detected/adpretrain_bridge/output/ahl_clip_base_official_effective_chmm_3cls_20260509/metrics/ahl_aligned_metrics_full.md`
- inference time: evaluator runs with `--benchmark-model-inference --device cuda`, reported as per-image feature-level AHL forward time.
