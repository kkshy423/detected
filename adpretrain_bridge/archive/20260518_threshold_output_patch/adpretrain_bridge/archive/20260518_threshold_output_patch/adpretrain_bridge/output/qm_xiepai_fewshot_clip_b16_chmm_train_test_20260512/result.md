# qm_xiepai Few-Shot CLIP-B16 + AHL + CHMM Train/Test

## First-Round Curve

| Stage | Policy | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Inference ms | Train N/A | Test N/A |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | adpretrain_eval_best_f1 | 0.3000 | 1.0000 | 0.4615 | 0.3000 | 0.2831 | 0.2227 | 5.9078 | 50/0 | 140/60 |
| S2 | adpretrain_eval_best_f1 | 0.8269 | 0.7167 | 0.7679 | 0.8700 | 0.8949 | 0.7928 | 4.0388 | 150/3 | 140/60 |
| S4 | adpretrain_eval_best_f1 | 0.8246 | 0.7833 | 0.8034 | 0.8850 | 0.8990 | 0.8124 | 6.6404 | 300/10 | 140/60 |
| S6 | adpretrain_eval_best_f1 | 0.8800 | 0.7333 | 0.8000 | 0.8900 | 0.9044 | 0.8257 | 5.8870 | 500/40 | 140/60 |
| S8 | adpretrain_eval_best_f1 | 0.8197 | 0.8333 | 0.8264 | 0.8950 | 0.9150 | 0.8338 | 5.8382 | 560/139 | 140/60 |

## Evaluation Policy

- No calibration split is used.
- AHL original code has no fixed classification threshold; it reports anomaly scores with AUC-ROC/AUC-PR.
- Primary Precision/Recall/F1 follow ADPretrain-style best-F1 on fixed test scores, matching the no-AHL evaluation standard.
- ADPretrain/projector is not fine-tuned; the existing CLIP-B16 official projected residual feature cache and CHMM cache are reused.

## Trend Judgement

- Precision S0 -> S8: 0.3000 -> 0.8197 (delta 0.5197).
- Recall S0 -> S8: 1.0000 -> 0.8333 (delta -0.1667).
- F1 S0 -> S8: 0.4615 -> 0.8264 (delta 0.3649).
- First usable stage under P/R/F1 >= 0.80/0.80/0.80: S8.
- First full-supervised switch stage under P/R/F1 >= 0.90/0.90/0.90: not reached.

## Files

- Summary JSON: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_train_test_20260512/metrics/fewshot_curve/fewshot_curve_summary.json`
- Summary CSV: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_train_test_20260512/metrics/fewshot_curve/fewshot_curve_summary.csv`
- Output root: `/ghome/huangjd/code/detected/adpretrain_bridge/output/qm_xiepai_fewshot_clip_b16_chmm_train_test_20260512`
