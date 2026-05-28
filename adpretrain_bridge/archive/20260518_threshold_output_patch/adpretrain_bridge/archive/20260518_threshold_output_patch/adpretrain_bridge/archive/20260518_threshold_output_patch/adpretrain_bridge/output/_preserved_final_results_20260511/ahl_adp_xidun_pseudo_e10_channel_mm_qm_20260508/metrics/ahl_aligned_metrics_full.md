# AHL Aligned Metrics

## Mean

| Class Count | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8207 | 0.8495 | 0.8360 | 0.8427 | 0.8762 | 0.9153 | N/A |

## Per Class

| Class | Accuracy | Acc>=98% | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms | Time<=50ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| models_qiumianxiepai_chmm | 0.8207 | FAIL | 0.8495 | 0.8360 | 0.8427 | 0.8762 | 0.9153 | N/A | N/A |

## Scope

- Alignable: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, optional inference time
- Not alignable: detection mAP, segmentation IoU

## Files

- JSON: `/ghome/huangjd/code/detected/adpretrain_bridge/output/ahl_adp_xidun_pseudo_e10_channel_mm_qm_20260508/metrics/ahl_aligned_metrics_full.json`
- Markdown: `/ghome/huangjd/code/detected/adpretrain_bridge/output/ahl_adp_xidun_pseudo_e10_channel_mm_qm_20260508/metrics/ahl_aligned_metrics_full.md`
