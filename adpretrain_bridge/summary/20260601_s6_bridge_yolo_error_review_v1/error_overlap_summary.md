# 20260601 S6 Bridge vs YOLO Error Review V1

- scope: S6 test set only
- no retraining, no threshold change, no alpha search
- bridge: ADP-AHL bridge v1, alpha=0.35, fixed threshold from `stage_aware_adp_ahl_bridge_v1`
- YOLO: existing `YOLO26s-eval` S6 threshold from fixed calibration policy
- test labels are used only to categorize errors

## Error Counts

| item | count |
|---|---:|
| bridge FP | 9 |
| bridge FN | 7 |
| YOLO FP | 20 |
| YOLO FN | 2 |
| bridge missed but YOLO hit | 6 |
| YOLO FP but bridge TN | 17 |
| common FN | 1 |
| common FP | 3 |
| bridge FP but YOLO TN | 6 |
| YOLO missed but bridge hit | 1 |

## Initial Judgment

- The two methods are complementary at S6: YOLO recovers bridge misses, while bridge rejects a non-trivial number of YOLO false positives.
- Practical reading: keep S6 as bridge default low-FP mode, and use YOLO as high-recall quality mode or supervised switch candidate.
- Common FN count is `1`; these samples need visual review because both methods miss them.

## Montages

- montage_bridge_missed_yolo_hit.jpg: `montage_bridge_missed_yolo_hit.jpg`
- montage_yolo_fp_bridge_tn.jpg: `montage_yolo_fp_bridge_tn.jpg`
- montage_common_fn.jpg: `montage_common_fn.jpg`
- montage_common_fp.jpg: `montage_common_fp.jpg`