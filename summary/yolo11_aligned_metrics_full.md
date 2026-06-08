# YOLO11 Aligned Metrics

## Mean

| Class Count | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | mAP50(B) | IoU | Time ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.6868 | 0.5940 | 0.8217 | 0.6475 | 0.6603 | 0.6185 | 0.7410 | 0.8043 | 33.2603 |

## Per Class

| Class | Accuracy | Acc>=98% | Precision | Recall | AUC-ROC | AUC-PR | mAP50(B) | mAP>=95% | IoU | IoU>=0.7 | Time ms | Time<=50ms |

| models__球面斜拍 | 0.9389 | FAIL | 0.8919 | 0.8250 | 0.9361 | 0.8977 | 0.8811 | FAIL | 0.8241 | PASS | 30.8311 | PASS |
| qiusai__models__底面检测__端面检测 | 0.8087 | FAIL | 0.7879 | 0.6341 | 0.7370 | 0.7183 | 0.6682 | FAIL | 0.8965 | PASS | 31.6842 | PASS |
| qiusai__models__球面俯拍 | 0.6667 | FAIL | 0.6667 | 1.0000 | 0.5014 | 0.7514 | 0.8958 | FAIL | 0.8955 | PASS | 33.9927 | PASS |
| yuanzhu__models__内孔中 | 0.6970 | FAIL | 0.5789 | 0.8462 | 0.7462 | 0.6660 | 0.7486 | FAIL | 0.5623 | FAIL | 31.9569 | PASS |
| yuanzhu__models__孔口 | 0.8043 | FAIL | 0.4545 | 0.6250 | 0.8092 | 0.4226 | 0.6170 | FAIL | 0.8552 | PASS | 35.0762 | PASS |
| yuanzhu__models__端面 | 0.2051 | FAIL | 0.1842 | 1.0000 | 0.2321 | 0.2553 | 0.6353 | FAIL | 0.7921 | PASS | 36.0205 | PASS |

## Scope

- Alignable: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, mAP50(B), IoU, inference time

## Files

- JSON: `/ghome/huangjd/code/detected/summary/yolo11_aligned_metrics_full.json`
- Markdown: `/ghome/huangjd/code/detected/summary/yolo11_aligned_metrics_full.md`
