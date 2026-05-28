# AHL Aligned Metrics

## Mean

| Class Count | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.9040 | 0.9166 | 0.9489 | 0.9317 | 0.8725 | 0.9677 | 10.6340 |

## Per Class

| Class | Accuracy | Acc>=98% | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms | Time<=50ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| models_clip_official_effective_chmm | 0.8511 | FAIL | 0.8889 | 0.8466 | 0.8672 | 0.9043 | 0.9274 | 11.6768 | PASS |
| qiusai_clip_official_effective_chmm | 0.9222 | FAIL | 0.9222 | 1.0000 | 0.9595 | 0.8443 | 0.9865 | 10.5457 | PASS |
| yuanzhu_clip_official_effective_chmm | 0.9388 | FAIL | 0.9387 | 1.0000 | 0.9684 | 0.8689 | 0.9894 | 9.6794 | PASS |

## Scope

- Alignable: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, optional inference time
- Not alignable: detection mAP, segmentation IoU

## Files

- JSON: `/ghome/huangjd/code/detected/adpretrain_bridge/output/threecls_threeway_20260509/metrics/clip_chmm/ahl_aligned_metrics_full.json`
- Markdown: `/ghome/huangjd/code/detected/adpretrain_bridge/output/threecls_threeway_20260509/metrics/clip_chmm/ahl_aligned_metrics_full.md`
