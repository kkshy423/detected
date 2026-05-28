# AHL Aligned Metrics

## Mean

| Class Count | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7781 | 0.7661 | 0.8836 | 0.8206 | 0.8254 | 0.8466 | N/A |

## Per Class

| Class | Accuracy | Acc>=98% | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms | Time<=50ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| models_qiumianxiepai | 0.7781 | FAIL | 0.7661 | 0.8836 | 0.8206 | 0.8254 | 0.8466 | N/A | N/A |

## Scope

- Alignable: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, optional inference time
- Not alignable: detection mAP, segmentation IoU

## Files

- JSON: `/ghome/huangjd/code/detected/adpretrain_bridge/output/ahl_adp_xidun_pseudo_e10_no_cudnn_models_qiumian_ascii_20260501/metrics/ahl_aligned_metrics_full.json`
- Markdown: `/ghome/huangjd/code/detected/adpretrain_bridge/output/ahl_adp_xidun_pseudo_e10_no_cudnn_models_qiumian_ascii_20260501/metrics/ahl_aligned_metrics_full.md`
