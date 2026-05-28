# qm_xiepai Few-Shot CLIP-B16 + AHL + CHMM Train/Test

## First-Round Curve

| Stage | Policy | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms | Train N/A | Test N/A |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | adpretrain_eval_best_f1 | 0.5254 | 0.5167 | 0.5210 | 0.7150 | 0.6831 | 0.5246 | 9.6924 | 50/0 | 140/60 |
| S1 | adpretrain_eval_best_f1 | 0.6724 | 0.6500 | 0.6610 | 0.8000 | 0.7846 | 0.6956 | 6.3768 | 100/1 | 140/60 |
| S2 | adpretrain_eval_best_f1 | 0.8400 | 0.7000 | 0.7636 | 0.8700 | 0.8736 | 0.7732 | 5.9608 | 150/3 | 140/60 |
| S3 | adpretrain_eval_best_f1 | 0.8704 | 0.7833 | 0.8246 | 0.9000 | 0.9194 | 0.8270 | 6.3975 | 200/5 | 140/60 |
| S4 | adpretrain_eval_best_f1 | 0.7879 | 0.8667 | 0.8254 | 0.8900 | 0.9279 | 0.8727 | 6.5100 | 300/10 | 140/60 |
| S5 | adpretrain_eval_best_f1 | 0.8679 | 0.7667 | 0.8142 | 0.8950 | 0.8940 | 0.8471 | 4.6821 | 400/20 | 140/60 |
| S6 | adpretrain_eval_best_f1 | 0.8824 | 0.7500 | 0.8108 | 0.8950 | 0.9280 | 0.8702 | 7.8726 | 500/40 | 140/60 |
| S7 | adpretrain_eval_best_f1 | 0.8393 | 0.7833 | 0.8103 | 0.8900 | 0.9187 | 0.8625 | 6.0340 | 560/80 | 140/60 |
| S8 | adpretrain_eval_best_f1 | 0.8136 | 0.8000 | 0.8067 | 0.8850 | 0.9224 | 0.8654 | 6.7572 | 560/139 | 140/60 |

## Evaluation Policy

- No calibration split is used.
- AHL original code has no fixed classification threshold; it reports anomaly scores with AUC-ROC/AUC-PR.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores, matching the no-AHL evaluation standard.
- ADPretrain/projector is not fine-tuned; the existing CLIP-B16 official projected residual feature cache and CHMM cache are reused.

## Trend Judgement

- Precision S0 -> S8: 0.5254 -> 0.8136 (delta 0.2881).
- Recall S0 -> S8: 0.5167 -> 0.8000 (delta 0.2833).
- F1 S0 -> S8: 0.5210 -> 0.8067 (delta 0.2857).
- First usable stage under P/R/F1 >= 0.80/0.80/0.80: S8.
- First full-supervised switch stage under P/R/F1 >= 0.90/0.90/0.90: not reached.

## Files

- Summary JSON: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_plain_train_test_20260512/metrics/fewshot_curve/fewshot_curve_summary.json`
- Summary CSV: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_plain_train_test_20260512/metrics/fewshot_curve/fewshot_curve_summary.csv`
- Output root: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_plain_train_test_20260512`
