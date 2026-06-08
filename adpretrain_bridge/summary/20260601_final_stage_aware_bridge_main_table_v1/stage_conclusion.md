# 20260601 Stage Conclusions

## S0-S2: Cold Start

- Use `ADP-only-DINO` as the cold-start mainline. It does not require AHL training and gives stable high-recall behavior: `R=0.9857`, `F1=0.7753`.
- `AHL-DINO` can be observed experimentally but is not the production mainline in this phase. Its early-stage false positives remain high and S0 does not yet have anomaly supervision.
- `YOLO26s-eval` is not trainable at S0 and remains an observation baseline at S1-S2. It is not the recommended production method in this phase.
- `ADP-AHL bridge v1` is `not_applicable` for S0-S2.

## S3-S4: Few-Shot Transition

- Keep `ADP-only-DINO` as the practical transition mainline because it preserves `R=0.9857` and does not add AHL-path latency.
- Track `AHL-DINO` as a few-shot observation baseline. It becomes more useful than in S0-S2, but it does not yet justify adding a bridge branch.
- `YOLO26s-eval` is still an observation baseline. Its recall is high, but precision and false positives are not yet competitive enough for switching.
- `ADP-AHL bridge v1` remains `not_applicable` for S3-S4.

## S5-S7: Stage-Aware Bridge Mainline

- Use `ADP-AHL bridge v1` with `alpha=0.35` as the mainline candidate.
- Test F1 is `0.8951 / 0.8873 / 0.8676` for S5/S6/S7, exceeding the corresponding ADP-only and AHL-only baselines.
- Calibration bootstrap stability supports this choice: bridge win-rate versus ADP-only is `1.00 / 0.97 / 0.97` for S5/S6/S7.
- Keep `ADP-only-DINO` as fallback because it is simpler and faster. Keep `AHL-DINO` as a complementary baseline rather than a standalone production decision path.

## S8: Late Stable Stage

- Use `ADP-AHL bridge v1` with `alpha=0.70` as the stage-aware bridge mainline candidate.
- On the fixed test set, bridge-alpha70 gives `P=0.8923`, `R=0.8286`, `F1=0.8593`, `FP=7`, `FN=12`.
- Bootstrap stability supports the bridge candidate: win-rate versus ADP-only is `0.84`, with mean recall `0.8340`.
- Keep `ADP-only-DINO` as the low-complexity fallback. It has slightly higher fixed-test recall (`0.8429`) and lower existing latency estimate, but lower F1 (`0.8429`) and more false positives (`11` versus `7`).

## YOLO Switch Window

- Treat S6 as the beginning of the supervised switch window: YOLO reaches `P=0.7727`, `R=0.9714`, `F1=0.8608`.
- Use YOLO as the supervised mainline at S7-S8: `F1=0.8759 / 0.8921`, with substantially lower measured latency than the AHL-path estimate.
- Keep `ADP-AHL bridge v1` as the anomaly-detection comparison line and as a deployable alternative when the supervised YOLO path is not yet trusted or requires additional production validation.

## Production Note

- All thresholds in the main table are selected from calibration data only. Test labels are used only to report held-out metrics.
- Bridge `Total_ms` currently reuses the AHL-path latency estimate. It is not a new end-to-end bridge latency measurement.
