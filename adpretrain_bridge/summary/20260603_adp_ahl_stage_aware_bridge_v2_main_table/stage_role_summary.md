# Stage Role Summary

## S0

- Use `ADP-only-DINO` for cold start. Bridge v2 is `not_applicable` because no AHL correction should be introduced before the early evidence phase.
- `AHL-DINO` is observation only and should not be described as an all-stage standalone solution.
- `YOLO26s-eval` is not trainable at S0.

## S1-S2

- Use `ADP-AHL bridge v2` as the early bridge mainline candidate with `alpha=0.70`.
- The early bridge improves over ADP-only on F1 and AUC-PR while keeping FN at zero in the fixed test report.
- This means AHL correction can start contributing before S5, but only through the ADP-AHL score bridge, not as standalone AHL.

## S3-S4

- Use `ADP-AHL bridge v2` as the early transition mainline candidate with `alpha=0.60`.
- S4 is the clearest early bridge gain: FP drops strongly while recall remains high.
- ADP-only remains the simple fallback; AHL-DINO remains an observation/complementary source.

## S5-S7

- Use `ADP-AHL bridge v2` as the small-sample transition mainline with `alpha=0.35`.
- Previous bootstrap check supports S5-S7 bridge stability: bridge win-rate versus ADP-only was `1.00 / 0.97 / 0.97`.
- The effective method is the score bridge. Do not present AHL-DINO alone as the production method.

## S8

- `ADP-AHL bridge v2` with `alpha=0.70` remains usable and gives a strong anomaly-detection comparison line.
- `YOLO26s-eval` has entered the fully supervised mainline at S8, with the highest F1 in this table.
- Keep bridge v2 as fallback/comparison when supervised deployment trust, retraining cadence, or labeling stability is not yet sufficient.
