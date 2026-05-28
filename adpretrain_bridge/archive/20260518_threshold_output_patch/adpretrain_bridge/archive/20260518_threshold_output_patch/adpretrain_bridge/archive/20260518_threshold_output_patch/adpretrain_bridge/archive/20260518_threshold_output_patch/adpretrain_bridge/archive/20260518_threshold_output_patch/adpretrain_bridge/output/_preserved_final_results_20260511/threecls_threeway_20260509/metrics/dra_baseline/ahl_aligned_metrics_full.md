# AHL Aligned Metrics

## Mean

| Class Count | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.9105 | 0.9424 | 0.9335 | 0.9349 | 0.9031 | 0.9760 | 9.1346 |

## Per Class

| Class | Accuracy | Acc>=98% | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms | Time<=50ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| models | 0.8693 | FAIL | 0.9620 | 0.8042 | 0.8761 | 0.9074 | 0.9443 | 7.5344 | PASS |
| qiusai | 0.9222 | FAIL | 0.9222 | 1.0000 | 0.9595 | 0.8415 | 0.9864 | 9.6463 | PASS |
| yuanzhu | 0.9402 | FAIL | 0.9431 | 0.9963 | 0.9690 | 0.9603 | 0.9974 | 10.2232 | PASS |

## Scope

- Alignable: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, optional inference time
- Not alignable: detection mAP, segmentation IoU

## Files

- JSON: `/ghome/huangjd/code/detected/adpretrain_bridge/output/threecls_threeway_20260509/metrics/dra_baseline/ahl_aligned_metrics_full.json`
- Markdown: `/ghome/huangjd/code/detected/adpretrain_bridge/output/threecls_threeway_20260509/metrics/dra_baseline/ahl_aligned_metrics_full.md`
