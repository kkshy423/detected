# AHL Aligned Metrics

## Mean

| Class Count | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7964 | 0.8720 | 0.7566 | 0.8102 | 0.8634 | 0.9021 | N/A |

## Per Class

| Class | Accuracy | Acc>=98% | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms | Time<=50ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| models_qiumianxiepai_mm | 0.7964 | FAIL | 0.8720 | 0.7566 | 0.8102 | 0.8634 | 0.9021 | N/A | N/A |

## Scope

- Alignable: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, optional inference time
- Not alignable: detection mAP, segmentation IoU

## Files

- JSON: `/ghome/huangjd/code/detected/adpretrain_bridge/output/ahl_adp_xidun_pseudo_e10_moment_match_qm_20260501/metrics/ahl_aligned_metrics_full.json`
- Markdown: `/ghome/huangjd/code/detected/adpretrain_bridge/output/ahl_adp_xidun_pseudo_e10_moment_match_qm_20260501/metrics/ahl_aligned_metrics_full.md`
