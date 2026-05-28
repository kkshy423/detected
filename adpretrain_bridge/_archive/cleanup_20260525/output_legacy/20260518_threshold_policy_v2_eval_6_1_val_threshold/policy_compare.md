# 20260518 Threshold Policy V2 Eval

Only existing scores/metrics are reused. No retraining or re-scoring is performed.

method stage policy status acc prec recall time_ms f1 tp fp tn fn auc_roc auc_pr source_policy
AHL_plain_ADPretrain S0 production_normal_p95 ok 0.4878 0.2727 0.0300 1965200.7917 0.0541 3 8 97 97 0.3633 0.4153 
AHL_plain_ADPretrain S0 strategy_stage_adaptive ok 0.4829 0.4853 0.9900 1965200.7917 0.6513 99 105  1 0.3633 0.4153 strategy_recall_first_r95
AHL_plain_ADPretrain S0 strategy_mild_stage_v2 ok 0.4244 0.4483 0.7800 1965200.7917 0.5693 78 96 9 22 0.3633 0.4153 mild_recall90_fpr20
AHL_plain_ADPretrain S0 strategy_balanced_f1 ok 0.4244 0.4483 0.7800 1965200.7917 0.5693 78 96 9 22 0.3633 0.4153 
AHL_plain_ADPretrain S1 production_normal_p95 ok 0.8439 0.9250 0.7400 2443304.5595 0.8222 74 6 99 26 0.9291 0.9140 
AHL_plain_ADPretrain S1 strategy_stage_adaptive ok 0.4927 0.4902 1.0000 2443304.5595 0.6579 100 104 1  0.9291 0.9140 strategy_recall_first_r95
AHL_plain_ADPretrain S1 strategy_mild_stage_v2 ok 0.8341 0.7661 0.9500 2443304.5595 0.8482 95 29 76 5 0.9291 0.9140 mild_recall90_fpr20
AHL_plain_ADPretrain S1 strategy_balanced_f1 ok 0.8439 0.9146 0.7500 2443304.5595 0.8242 75 7 98 25 0.9291 0.9140 
AHL_plain_ADPretrain S2 production_normal_p95 ok 0.8683 0.9195 0.8000 2690698.2174 0.8556 80 7 98 20 0.9296 0.9257 
AHL_plain_ADPretrain S2 strategy_stage_adaptive ok 0.4878 0.4878 1.0000 2690698.2174 0.6557 100 105   0.9296 0.9257 strategy_recall_first_r95
AHL_plain_ADPretrain S2 strategy_mild_stage_v2 ok 0.8634 0.8158 0.9300 2690698.2174 0.8692 93 21 84 7 0.9296 0.9257 mild_recall90_fpr20
AHL_plain_ADPretrain S2 strategy_balanced_f1 ok 0.8732 0.9302 0.8000 2690698.2174 0.8602 80 6 99 20 0.9296 0.9257 
AHL_plain_ADPretrain S3 production_normal_p95 ok 0.8732 0.9302 0.8000 3144433.1799 0.8602 80 6 99 20 0.9274 0.9218 
AHL_plain_ADPretrain S3 strategy_stage_adaptive ok 0.4927 0.4902 1.0000 3144433.1799 0.6579 100 104 1  0.9274 0.9218 strategy_recall_first_r95
AHL_plain_ADPretrain S3 strategy_mild_stage_v2 ok 0.8683 0.9101 0.8100 3144433.1799 0.8571 81 8 97 19 0.9274 0.9218 mild_recall85_fpr10
AHL_plain_ADPretrain S3 strategy_balanced_f1 ok 0.8732 0.9302 0.8000 3144433.1799 0.8602 80 6 99 20 0.9274 0.9218 
AHL_plain_ADPretrain S4 production_normal_p95 ok 0.8585 0.9176 0.7800 3565180.9236 0.8432 78 7 98 22 0.9120 0.9174 
AHL_plain_ADPretrain S4 strategy_stage_adaptive ok 0.4878 0.4878 1.0000 3565180.9236 0.6557 100 105   0.9120 0.9174 strategy_recall_first_r95
AHL_plain_ADPretrain S4 strategy_mild_stage_v2 ok 0.8683 0.8763 0.8500 3565180.9236 0.8629 85 12 93 15 0.9120 0.9174 mild_recall85_fpr10
AHL_plain_ADPretrain S4 strategy_balanced_f1 ok 0.8488 0.9367 0.7400 3565180.9236 0.8268 74 5 100 26 0.9120 0.9174 
AHL_plain_ADPretrain S5 production_normal_p95 ok 0.8683 0.9195 0.8000 4055095.6721 0.8556 80 7 98 20 0.9148 0.9167 
AHL_plain_ADPretrain S5 strategy_stage_adaptive ok 0.4927 0.4902 1.0000 4055095.6721 0.6579 100 104 1  0.9148 0.9167 strategy_recall_first_r90
AHL_plain_ADPretrain S5 strategy_mild_stage_v2 ok 0.8390 0.8317 0.8400 4055095.6721 0.8358 84 17 88 16 0.9148 0.9167 mild_recall85_fpr10
AHL_plain_ADPretrain S5 strategy_balanced_f1 ok 0.8683 0.9101 0.8100 4055095.6721 0.8571 81 8 97 19 0.9148 0.9167 
AHL_plain_ADPretrain S6 production_normal_p95 ok 0.8732 0.9111 0.8200 4741183.8636 0.8632 82 8 97 18 0.9342 0.9268 
AHL_plain_ADPretrain S6 strategy_stage_adaptive ok 0.4927 0.4902 1.0000 4741183.8636 0.6579 100 104 1  0.9342 0.9268 strategy_recall_first_r90
AHL_plain_ADPretrain S6 strategy_mild_stage_v2 ok 0.8732 0.8776 0.8600 4741183.8636 0.8687 86 12 93 14 0.9342 0.9268 mild_recall85_fpr10
AHL_plain_ADPretrain S6 strategy_balanced_f1 ok 0.8146 0.9559 0.6500 4741183.8636 0.7738 65 3 102 35 0.9342 0.9268 
AHL_plain_ADPretrain S7 production_normal_p95 ok 0.8927 0.9239 0.8500 4791788.7287 0.8854 85 7 98 15 0.9396 0.9321 
AHL_plain_ADPretrain S7 strategy_stage_adaptive ok 0.8927 0.9239 0.8500 4791788.7287 0.8854 85 7 98 15 0.9396 0.9321 strategy_balanced_f1
AHL_plain_ADPretrain S7 strategy_mild_stage_v2 ok 0.8927 0.9239 0.8500 4791788.7287 0.8854 85 7 98 15 0.9396 0.9321 balanced_f1_late
AHL_plain_ADPretrain S7 strategy_balanced_f1 ok 0.8927 0.9239 0.8500 4791788.7287 0.8854 85 7 98 15 0.9396 0.9321 
AHL_plain_ADPretrain S8 production_normal_p95 ok 0.8683 0.9195 0.8000 4923044.8895 0.8556 80 7 98 20 0.9080 0.9112 
AHL_plain_ADPretrain S8 strategy_stage_adaptive ok 0.8683 0.9195 0.8000 4923044.8895 0.8556 80 7 98 20 0.9080 0.9112 strategy_balanced_f1
AHL_plain_ADPretrain S8 strategy_mild_stage_v2 ok 0.8683 0.9195 0.8000 4923044.8895 0.8556 80 7 98 20 0.9080 0.9112 balanced_f1_late
AHL_plain_ADPretrain S8 strategy_balanced_f1 ok 0.8683 0.9195 0.8000 4923044.8895 0.8556 80 7 98 20 0.9080 0.9112 
YOLO26s_cls S0 ALL skipped            
YOLO26s_cls S1 production_normal_p95 ok 0.6829 0.9070 0.3900 39.0169 0.5455 39 4 101 61 0.7846 0.8056 
YOLO26s_cls S1 strategy_stage_adaptive ok 0.4976 0.4926 1.0000 39.0169 0.6601 100 103 2  0.7846 0.8056 strategy_recall_first_r95
YOLO26s_cls S1 strategy_mild_stage_v2 ok 0.6146 0.5660 0.9000 39.0169 0.6950 90 69 36 10 0.7846 0.8056 mild_recall90_fpr20
YOLO26s_cls S1 strategy_balanced_f1 ok 0.6537 0.9143 0.3200 39.0169 0.4741 32 3 102 68 0.7846 0.8056 
YOLO26s_cls S2 production_normal_p95 ok 0.8341 0.9459 0.7000 40.3789 0.8046 70 4 101 30 0.9349 0.9432 
YOLO26s_cls S2 strategy_stage_adaptive ok 0.5024 0.4950 1.0000 40.3789 0.6623 100 102 3  0.9349 0.9432 strategy_recall_first_r95
YOLO26s_cls S2 strategy_mild_stage_v2 ok 0.8293 0.7686 0.9300 40.3789 0.8416 93 28 77 7 0.9349 0.9432 mild_recall90_fpr20
YOLO26s_cls S2 strategy_balanced_f1 ok 0.7756 0.9655 0.5600 40.3789 0.7089 56 2 103 44 0.9349 0.9432 
YOLO26s_cls S3 production_normal_p95 ok 0.8195 0.8987 0.7100 43.7902 0.7933 71 8 97 29 0.9169 0.9289 
YOLO26s_cls S3 strategy_stage_adaptive ok 0.4878 0.4878 1.0000 43.7902 0.6557 100 105   0.9169 0.9289 strategy_recall_first_r95
YOLO26s_cls S3 strategy_mild_stage_v2 ok 0.8293 0.9221 0.7100 43.7902 0.8023 71 6 99 29 0.9169 0.9289 mild_recall85_fpr10
YOLO26s_cls S3 strategy_balanced_f1 ok 0.7659 1.0000 0.5200 43.7902 0.6842 52  105 48 0.9169 0.9289 
YOLO26s_cls S5 production_normal_p95 ok 0.8634 0.9500 0.7600 30.1234 0.8444 76 4 101 24 0.9327 0.9460 
YOLO26s_cls S5 strategy_stage_adaptive ok 0.4976 0.4926 1.0000 30.1234 0.6601 100 103 2  0.9327 0.9460 strategy_recall_first_r90
YOLO26s_cls S5 strategy_mild_stage_v2 ok 0.8537 0.9861 0.7100 30.1234 0.8256 71 1 104 29 0.9327 0.9460 mild_recall85_fpr10
YOLO26s_cls S5 strategy_balanced_f1 ok 0.8146 1.0000 0.6200 30.1234 0.7654 62  105 38 0.9327 0.9460 
YOLO26s_cls S6 production_normal_p95 ok 0.8683 0.9398 0.7800 31.2349 0.8525 78 5 100 22 0.9277 0.9444 
YOLO26s_cls S6 strategy_stage_adaptive ok 0.4878 0.4878 1.0000 31.2349 0.6557 100 105   0.9277 0.9444 strategy_recall_first_r90
YOLO26s_cls S6 strategy_mild_stage_v2 ok 0.8439 1.0000 0.6800 31.2349 0.8095 68  105 32 0.9277 0.9444 mild_recall85_fpr10
YOLO26s_cls S6 strategy_balanced_f1 ok 0.8439 1.0000 0.6800 31.2349 0.8095 68  105 32 0.9277 0.9444 
YOLO26s_cls S7 production_normal_p95 ok 0.8780 0.9747 0.7700 33.6982 0.8603 77 2 103 23 0.9448 0.9601 
YOLO26s_cls S7 strategy_stage_adaptive ok 0.8634 1.0000 0.7200 33.6982 0.8372 72  105 28 0.9448 0.9601 strategy_balanced_f1
YOLO26s_cls S7 strategy_mild_stage_v2 ok 0.8634 1.0000 0.7200 33.6982 0.8372 72  105 28 0.9448 0.9601 balanced_f1_late
YOLO26s_cls S7 strategy_balanced_f1 ok 0.8634 1.0000 0.7200 33.6982 0.8372 72  105 28 0.9448 0.9601 
YOLO26s_cls S8 production_normal_p95 ok 0.9073 0.9263 0.8800 73.2833 0.9026 88 7 98 12 0.9664 0.9709 
YOLO26s_cls S8 strategy_stage_adaptive ok 0.8732 1.0000 0.7400 73.2833 0.8506 74  105 26 0.9664 0.9709 strategy_balanced_f1
YOLO26s_cls S8 strategy_mild_stage_v2 ok 0.8732 1.0000 0.7400 73.2833 0.8506 74  105 26 0.9664 0.9709 balanced_f1_late
YOLO26s_cls S8 strategy_balanced_f1 ok 0.8732 1.0000 0.7400 73.2833 0.8506 74  105 26 0.9664 0.9709 
YOLO26s_cls S0 ALL missing            
YOLO26s_cls S4 production_normal_p95 ok 0.8488 1.0000 0.6900 82.3443 0.8166 69  105 31 0.9373 0.9516 
YOLO26s_cls S4 strategy_stage_adaptive ok 0.4878 0.4878 1.0000 82.3443 0.6557 100 105   0.9373 0.9516 strategy_recall_first_r95
YOLO26s_cls S4 strategy_mild_stage_v2 ok 0.8683 1.0000 0.7300 82.3443 0.8439 73  105 27 0.9373 0.9516 mild_recall85_fpr10
YOLO26s_cls S4 strategy_balanced_f1 ok 0.8049 1.0000 0.6000 82.3443 0.7500 60  105 40 0.9373 0.9516 
