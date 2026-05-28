# AHL Aligned Metrics

## Mean

| Class Count | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8784 | 0.9027 | 0.8836 | 0.8930 | 0.9191 | 0.9288 | N/A |

## Per Class

| Class | Accuracy | Acc>=98% | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms | Time<=50ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| models_qiumianxiepai_clip_official_img_angle_chmm | 0.8784 | FAIL | 0.9027 | 0.8836 | 0.8930 | 0.9191 | 0.9288 | N/A | N/A |

## Scope

- Alignable: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, optional inference time
- Not alignable: detection mAP, segmentation IoU

## Files

- JSON: `/ghome/huangjd/code/detected/adpretrain_bridge/output/ahl_clip_base_official_img_angle_chmm_qm_20260508/metrics/ahl_aligned_metrics_full.json`
- Markdown: `/ghome/huangjd/code/detected/adpretrain_bridge/output/ahl_clip_base_official_img_angle_chmm_qm_20260508/metrics/ahl_aligned_metrics_full.md`
