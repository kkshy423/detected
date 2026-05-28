# AHL Aligned Metrics

## Mean

| Class Count | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.9033 | 0.8957 | 0.9779 | 0.9342 | 0.8958 | 0.9701 | 10.2101 |

## Per Class

| Class | Accuracy | Acc>=98% | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms | Time<=50ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| models_clip_official_effective_plain | 0.8328 | FAIL | 0.7991 | 0.9471 | 0.8668 | 0.9099 | 0.9292 | 9.7428 | PASS |
| qiusai_clip_official_effective_plain | 0.9222 | FAIL | 0.9222 | 1.0000 | 0.9595 | 0.8292 | 0.9851 | 9.9331 | PASS |
| yuanzhu_clip_official_effective_plain | 0.9548 | FAIL | 0.9659 | 0.9866 | 0.9761 | 0.9481 | 0.9961 | 10.9543 | PASS |

## Scope

- Alignable: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, optional inference time
- Not alignable: detection mAP, segmentation IoU

## Files

- JSON: `/ghome/huangjd/code/detected/adpretrain_bridge/output/threecls_threeway_20260509/metrics/clip_plain/ahl_aligned_metrics_full.json`
- Markdown: `/ghome/huangjd/code/detected/adpretrain_bridge/output/threecls_threeway_20260509/metrics/clip_plain/ahl_aligned_metrics_full.md`
