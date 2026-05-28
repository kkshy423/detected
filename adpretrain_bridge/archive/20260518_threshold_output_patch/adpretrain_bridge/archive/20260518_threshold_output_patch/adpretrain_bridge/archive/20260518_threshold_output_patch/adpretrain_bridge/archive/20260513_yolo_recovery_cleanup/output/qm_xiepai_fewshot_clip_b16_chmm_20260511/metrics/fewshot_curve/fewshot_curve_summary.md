# qm_xiepai Few-Shot CLIP-B16 + AHL + CHMM

## First-Round Curve

| Stage | Policy | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms | Calib N/A | Test N/A |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | normal_p97_5 | 0.8667 | 0.2167 | 0.3467 | 0.7550 | 0.7888 | 0.6907 | 9.6401 | 10/0 | 140/60 |
| S2 | normal_p97_5 | 0.8462 | 0.7333 | 0.7857 | 0.8800 | 0.9023 | 0.8014 | 5.8732 | 30/0 | 140/60 |
| S4 | calib_best_f1 | 0.7222 | 0.8667 | 0.7879 | 0.8600 | 0.9345 | 0.8534 | 10.3424 | 60/2 | 140/60 |
| S6 | calib_best_f1 | 0.8571 | 0.6000 | 0.7059 | 0.8500 | 0.8949 | 0.8060 | 7.8507 | 100/8 | 140/60 |
| S8 | calib_best_f1 | 0.7167 | 0.7167 | 0.7167 | 0.8300 | 0.8773 | 0.7751 | 5.8067 | 112/28 | 140/60 |

## Trend Judgement

- Precision S0 -> S8: 0.8667 -> 0.7167 (delta -0.1500).
- Recall S0 -> S8: 0.2167 -> 0.7167 (delta 0.5000).
- F1 S0 -> S8: 0.3467 -> 0.7167 (delta 0.3700).
- First usable stage under P/R/F1 >= 0.80/0.80/0.80: not reached.
- First full-supervised switch stage under P/R/F1 >= 0.90/0.90/0.90: not reached.
- First diminishing-return stage by F1 gain < 0.02: S4.

## Files

- Summary JSON: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511/metrics/fewshot_curve/fewshot_curve_summary.json`
- Summary CSV: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511/metrics/fewshot_curve/fewshot_curve_summary.csv`
- Output root: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_20260511`
