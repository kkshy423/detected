# AHL Aligned Metrics

## Mean

| Class Count | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.8880 | 0.9040 | 0.9193 | 0.9069 | 0.9340 | 0.9653 | 12.6147 |

## Per Class

| Class | Accuracy | Acc>=98% | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference ms | Time<=50ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| models__球面斜拍 | 0.8663 | FAIL | 0.8796 | 0.8889 | 0.8842 | 0.9420 | 0.9621 | 13.3860 | PASS |
| qiusai__models__底面检测__端面检测 | 0.7546 | FAIL | 0.7471 | 1.0000 | 0.8553 | 0.8512 | 0.9482 | 11.0852 | PASS |
| qiusai__models__球面俯拍 | 0.9337 | FAIL | 0.9607 | 0.9661 | 0.9634 | 0.9227 | 0.9910 | 11.7827 | PASS |
| yuanzhu__models__内孔中 | 0.9726 | FAIL | 1.0000 | 0.9623 | 0.9808 | 0.9943 | 0.9980 | 15.8533 | PASS |
| yuanzhu__models__孔口 | 0.8551 | FAIL | 0.9200 | 0.7419 | 0.8214 | 0.9075 | 0.9089 | 13.1665 | PASS |
| yuanzhu__models__端面 | 0.9455 | FAIL | 0.9167 | 0.9565 | 0.9362 | 0.9864 | 0.9836 | 10.4143 | PASS |

## Scope

- Alignable: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, optional inference time
- Not alignable: detection mAP, segmentation IoU

## Files

- JSON: `/ghome/huangjd/code/detected/summary/ahl_aligned_metrics_full.json`
- Markdown: `/ghome/huangjd/code/detected/summary/ahl_aligned_metrics_full.md`
