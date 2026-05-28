# 20260518 Policy Compare Eval 6_1 Val Threshold

???????? scores/metrics ???????????production_normal_p95 ????????strategy_stage_adaptive ?????????

method stage policy status acc prec recall time_ms f1 tp fp tn fn auc_roc auc_pr
AHL_plain_ADPretrain S0 production_normal_p95 ok 0.4878 0.2727 0.0300 1965200.7917 0.0541 3 8 97 97 0.3633 0.4153
AHL_plain_ADPretrain S0 strategy_stage_adaptive ok 0.4829 0.4853 0.9900 1965200.7917 0.6513 99 105 0 1 0.3633 0.4153
AHL_plain_ADPretrain S1 production_normal_p95 ok 0.8439 0.9250 0.7400 2443304.5595 0.8222 74 6 99 26 0.9291 0.9140
AHL_plain_ADPretrain S1 strategy_stage_adaptive ok 0.4927 0.4902 1.0000 2443304.5595 0.6579 100 104 1 0 0.9291 0.9140
AHL_plain_ADPretrain S2 production_normal_p95 ok 0.8683 0.9195 0.8000 2690698.2174 0.8556 80 7 98 20 0.9296 0.9257
AHL_plain_ADPretrain S2 strategy_stage_adaptive ok 0.4878 0.4878 1.0000 2690698.2174 0.6557 100 105 0 0 0.9296 0.9257
AHL_plain_ADPretrain S3 production_normal_p95 ok 0.8732 0.9302 0.8000 3144433.1799 0.8602 80 6 99 20 0.9274 0.9218
AHL_plain_ADPretrain S3 strategy_stage_adaptive ok 0.4927 0.4902 1.0000 3144433.1799 0.6579 100 104 1 0 0.9274 0.9218
AHL_plain_ADPretrain S4 production_normal_p95 ok 0.8585 0.9176 0.7800 3565180.9236 0.8432 78 7 98 22 0.9120 0.9174
AHL_plain_ADPretrain S4 strategy_stage_adaptive ok 0.4878 0.4878 1.0000 3565180.9236 0.6557 100 105 0 0 0.9120 0.9174
AHL_plain_ADPretrain S5 production_normal_p95 ok 0.8683 0.9195 0.8000 4055095.6721 0.8556 80 7 98 20 0.9148 0.9167
AHL_plain_ADPretrain S5 strategy_stage_adaptive ok 0.4927 0.4902 1.0000 4055095.6721 0.6579 100 104 1 0 0.9148 0.9167
AHL_plain_ADPretrain S6 production_normal_p95 ok 0.8732 0.9111 0.8200 4741183.8636 0.8632 82 8 97 18 0.9342 0.9268
AHL_plain_ADPretrain S6 strategy_stage_adaptive ok 0.4927 0.4902 1.0000 4741183.8636 0.6579 100 104 1 0 0.9342 0.9268
AHL_plain_ADPretrain S7 production_normal_p95 ok 0.8927 0.9239 0.8500 4791788.7287 0.8854 85 7 98 15 0.9396 0.9321
AHL_plain_ADPretrain S7 strategy_stage_adaptive ok 0.8927 0.9239 0.8500 4791788.7287 0.8854 85 7 98 15 0.9396 0.9321
AHL_plain_ADPretrain S8 production_normal_p95 ok 0.8683 0.9195 0.8000 4923044.8895 0.8556 80 7 98 20 0.9080 0.9112
AHL_plain_ADPretrain S8 strategy_stage_adaptive ok 0.8683 0.9195 0.8000 4923044.8895 0.8556 80 7 98 20 0.9080 0.9112
YOLO26s_cls S0 ALL skipped           
YOLO26s_cls S1 production_normal_p95 ok 0.6829 0.9070 0.3900 39.0169 0.5455 39 4 101 61 0.7846 0.8056
YOLO26s_cls S1 strategy_stage_adaptive ok 0.4976 0.4926 1.0000 39.0169 0.6601 100 103 2 0 0.7846 0.8056
YOLO26s_cls S2 production_normal_p95 ok 0.8341 0.9459 0.7000 40.3789 0.8046 70 4 101 30 0.9349 0.9432
YOLO26s_cls S2 strategy_stage_adaptive ok 0.5024 0.4950 1.0000 40.3789 0.6623 100 102 3 0 0.9349 0.9432
YOLO26s_cls S3 production_normal_p95 ok 0.8195 0.8987 0.7100 43.7902 0.7933 71 8 97 29 0.9169 0.9289
YOLO26s_cls S3 strategy_stage_adaptive ok 0.4878 0.4878 1.0000 43.7902 0.6557 100 105 0 0 0.9169 0.9289
YOLO26s_cls S5 production_normal_p95 ok 0.8634 0.9500 0.7600 30.1234 0.8444 76 4 101 24 0.9327 0.9460
YOLO26s_cls S5 strategy_stage_adaptive ok 0.4976 0.4926 1.0000 30.1234 0.6601 100 103 2 0 0.9327 0.9460
YOLO26s_cls S6 production_normal_p95 ok 0.8683 0.9398 0.7800 31.2349 0.8525 78 5 100 22 0.9277 0.9444
YOLO26s_cls S6 strategy_stage_adaptive ok 0.4878 0.4878 1.0000 31.2349 0.6557 100 105 0 0 0.9277 0.9444
YOLO26s_cls S7 production_normal_p95 ok 0.8780 0.9747 0.7700 33.6982 0.8603 77 2 103 23 0.9448 0.9601
YOLO26s_cls S7 strategy_stage_adaptive ok 0.8634 1.0000 0.7200 33.6982 0.8372 72 0 105 28 0.9448 0.9601
YOLO26s_cls S8 production_normal_p95 ok 0.9073 0.9263 0.8800 73.2833 0.9026 88 7 98 12 0.9664 0.9709
YOLO26s_cls S8 strategy_stage_adaptive ok 0.8732 1.0000 0.7400 73.2833 0.8506 74 0 105 26 0.9664 0.9709
YOLO26s_cls S0 ALL missing           
YOLO26s_cls S4 production_normal_p95 ok 0.8488 1.0000 0.6900 82.3443 0.8166 69 0 105 31 0.9373 0.9516
YOLO26s_cls S4 strategy_stage_adaptive ok 0.4878 0.4878 1.0000 82.3443 0.6557 100 105 0 0 0.9373 0.9516
