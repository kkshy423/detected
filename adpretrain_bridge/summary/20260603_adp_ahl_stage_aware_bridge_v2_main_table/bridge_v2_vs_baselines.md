# Bridge V2 vs Baselines

| Stage | Bridge alpha | Bridge F1 | ADP F1 | AHL F1 | YOLO F1 | Bridge FP/FN | ADP FP/FN | YOLO FP/FN | Reading |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| S0 |  |  | 0.7753 | 0.5536 |  | / | 39/1 | / | bridge not applicable; ADP-only cold start |
| S1 | 0.7000 | 0.8092 | 0.7753 | 0.7159 | 0.4389 | 33/0 | 39/1 | 179/0 | early bridge improves ADP-only; AHL correction is useful only inside bridge |
| S2 | 0.7000 | 0.8000 | 0.7753 | 0.5317 | 0.5877 | 35/0 | 39/1 | 79/8 | early bridge improves ADP-only; AHL correction is useful only inside bridge |
| S3 | 0.6000 | 0.8187 | 0.7931 | 0.7665 | 0.4962 | 31/0 | 35/1 | 130/4 | early bridge improves ADP-only; AHL correction is useful only inside bridge |
| S4 | 0.6000 | 0.9139 | 0.7931 | 0.7602 | 0.6667 | 12/1 | 35/1 | 60/5 | early bridge improves ADP-only; AHL correction is useful only inside bridge |
| S5 | 0.3500 | 0.8951 | 0.7931 | 0.7898 | 0.6233 | 9/6 | 35/1 | 78/3 | bridge is small-sample transition mainline; YOLO approaches switch window by S6-S7 |
| S6 | 0.3500 | 0.8873 | 0.7931 | 0.7952 | 0.8608 | 9/7 | 35/1 | 20/2 | bridge is small-sample transition mainline; YOLO approaches switch window by S6-S7 |
| S7 | 0.3500 | 0.8676 | 0.8429 | 0.7899 | 0.8759 | 7/11 | 11/11 | 7/10 | bridge is small-sample transition mainline; YOLO approaches switch window by S6-S7 |
| S8 | 0.7000 | 0.8593 | 0.8429 | 0.8235 | 0.8921 | 7/12 | 11/11 | 7/8 | YOLO is supervised mainline; bridge remains usable fallback/comparison |

## Key Points

- Bridge v2 is not a claim that `AHL-DINO` alone is effective across all stages.
- The useful signal is the ADP-AHL score-level combination with stage-aware alpha.
- S1-S4 show early correction value; S5-S7 define the few-shot transition bridge; S8 remains bridge-usable but YOLO is the supervised mainline.
