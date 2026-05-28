# ahl_clip_base_official_effective_chmm_6cls_20260508

Status: submitted PBS, waiting for completion.

- Job ID: `204052.Ghead`
- PBS job name: `clip_off_6cls`
- launcher: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_run/run_clip_base_official_effective_chmm_6cls_20260508.sh`
- PBS: `/ghome/huangjd/code/detected/adpretrain_bridge/pbs/generated_submit/clip_base_official_effective_chmm_6cls_20260508.pbs`
- plain cache: `/gdata1/huangjd/data/xidun_mvtec_format_adpretrain_clip_base_official_effective_6cls_20260508`
- CHMM cache: `/gdata1/huangjd/data/xidun_mvtec_format_adpretrain_clip_base_official_effective_chmm_6cls_20260508`
- metrics target: `/ghome/huangjd/code/detected/adpretrain_bridge/output/ahl_clip_base_official_effective_chmm_6cls_20260508/metrics/ahl_aligned_metrics_full.md`


## 2026-05-09 result check

Status: exited before full 6-class metrics; partial metrics generated.

- Completed classes: 2/6
- Failed class: `qiusai_models_qiumianfupai_clip_official_effective_chmm`
- Error: `ValueError: factorial() not defined for negative values`
- Partial metrics: `/ghome/huangjd/code/detected/adpretrain_bridge/output/ahl_clip_base_official_effective_chmm_6cls_20260508/metrics_partial/ahl_aligned_metrics_full.md`
- Cause: AHL effective cluster count became `<2` under default `cluster_num=3`, so episode calculation hit `factorial(m - 2)`.
