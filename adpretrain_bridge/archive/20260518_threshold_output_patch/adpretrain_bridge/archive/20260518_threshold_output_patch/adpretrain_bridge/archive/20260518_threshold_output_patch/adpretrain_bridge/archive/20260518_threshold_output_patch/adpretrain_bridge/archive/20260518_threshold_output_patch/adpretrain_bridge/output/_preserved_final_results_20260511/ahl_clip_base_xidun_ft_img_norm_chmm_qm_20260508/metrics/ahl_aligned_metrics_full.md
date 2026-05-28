# AHL Aligned Metrics

## Mean

| Class Count | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8541 | 0.9123 | 0.8254 | 0.8667 | 0.9104 | 0.9307 | N/A |

## Per Class

| Class | Accuracy | Acc>=98% | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms | Time<=50ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| models_qiumianxiepai_clip_xidun_ft_img_norm_chmm | 0.8541 | FAIL | 0.9123 | 0.8254 | 0.8667 | 0.9104 | 0.9307 | N/A | N/A |

## Scope

- Alignable: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, optional inference time
- Not alignable: detection mAP, segmentation IoU

## Files

- JSON: `/ghome/huangjd/code/detected/adpretrain_bridge/output/ahl_clip_base_xidun_ft_img_norm_chmm_qm_20260508/metrics/ahl_aligned_metrics_full.json`
- Markdown: `/ghome/huangjd/code/detected/adpretrain_bridge/output/ahl_clip_base_xidun_ft_img_norm_chmm_qm_20260508/metrics/ahl_aligned_metrics_full.md`
