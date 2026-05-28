# 20260518 Threshold Policy V2 Eval

Only existing scores/metrics are reused. No retraining or re-scoring is performed.

method stage policy status acc prec recall time_ms f1 tp fp tn fn auc_roc auc_pr source_policy
AHL_plain_ADPretrain S0 production_normal_p95 ok 0.6023 0.1000 0.0380 3153857.0463 0.0550 3 27 153 76 0.3244 0.2280 
AHL_plain_ADPretrain S0 strategy_stage_adaptive ok 0.3050 0.3050 1.0000 3153857.0463 0.4675 79 180   0.3244 0.2280 strategy_recall_first_r95
AHL_plain_ADPretrain S0 strategy_mild_stage_v2 ok 0.3050 0.3050 1.0000 3153857.0463 0.4675 79 180   0.3244 0.2280 mild_recall90_fpr20
AHL_plain_ADPretrain S0 strategy_stage_v3 ok 0.3012 0.2992 0.9620 3153857.0463 0.4565 76 178 2 3 0.3244 0.2280 recall_at_95_highest_threshold
AHL_plain_ADPretrain S0 recall_at_95_highest_threshold ok 0.3012 0.2992 0.9620 3153857.0463 0.4565 76 178 2 3 0.3244 0.2280 
AHL_plain_ADPretrain S0 recall_at_90_highest_threshold ok 0.2973 0.2964 0.9494 3153857.0463 0.4518 75 178 2 4 0.3244 0.2280 
AHL_plain_ADPretrain S0 strategy_balanced_f1 ok 0.3050 0.3050 1.0000 3153857.0463 0.4675 79 180   0.3244 0.2280 
AHL_plain_ADPretrain S1 production_normal_p95 ok 0.8224 0.8667 0.4937 3668855.6516 0.6290 39 6 174 40 0.8771 0.8125 
AHL_plain_ADPretrain S1 strategy_stage_adaptive ok 0.3050 0.3050 1.0000 3668855.6516 0.4675 79 180   0.8771 0.8125 strategy_recall_first_r95
AHL_plain_ADPretrain S1 strategy_mild_stage_v2 ok 0.6602 0.4706 0.9114 3668855.6516 0.6207 72 81 99 7 0.8771 0.8125 mild_recall90_fpr20
AHL_plain_ADPretrain S1 strategy_stage_v3 ok 0.6602 0.4706 0.9114 3668855.6516 0.6207 72 81 99 7 0.8771 0.8125 recall_at_95_highest_threshold
AHL_plain_ADPretrain S1 recall_at_95_highest_threshold ok 0.6602 0.4706 0.9114 3668855.6516 0.6207 72 81 99 7 0.8771 0.8125 
AHL_plain_ADPretrain S1 recall_at_90_highest_threshold ok 0.8571 0.7917 0.7215 3668855.6516 0.7550 57 15 165 22 0.8771 0.8125 
AHL_plain_ADPretrain S1 strategy_balanced_f1 ok 0.8456 0.8679 0.5823 3668855.6516 0.6970 46 7 173 33 0.8771 0.8125 
AHL_plain_ADPretrain S2 production_normal_p95 ok 0.8649 0.9074 0.6203 3876124.4129 0.7368 49 5 175 30 0.9432 0.8869 
AHL_plain_ADPretrain S2 strategy_stage_adaptive ok 0.3089 0.3062 1.0000 3876124.4129 0.4688 79 179 1  0.9432 0.8869 strategy_recall_first_r95
AHL_plain_ADPretrain S2 strategy_mild_stage_v2 ok 0.8726 0.7556 0.8608 3876124.4129 0.8047 68 22 158 11 0.9432 0.8869 mild_recall90_fpr20
AHL_plain_ADPretrain S2 strategy_stage_v3 ok 0.8726 0.7556 0.8608 3876124.4129 0.8047 68 22 158 11 0.9432 0.8869 recall_at_95_highest_threshold
AHL_plain_ADPretrain S2 recall_at_95_highest_threshold ok 0.8726 0.7556 0.8608 3876124.4129 0.8047 68 22 158 11 0.9432 0.8869 
AHL_plain_ADPretrain S2 recall_at_90_highest_threshold ok 0.8919 0.8806 0.7468 3876124.4129 0.8082 59 8 172 20 0.9432 0.8869 
AHL_plain_ADPretrain S2 strategy_balanced_f1 ok 0.8726 0.8966 0.6582 3876124.4129 0.7591 52 6 174 27 0.9432 0.8869 
AHL_plain_ADPretrain S3 production_normal_p95 ok 0.8764 0.9273 0.6456 3250976.7529 0.7612 51 4 176 28 0.9244 0.8793 
AHL_plain_ADPretrain S3 strategy_stage_adaptive ok 0.3050 0.3050 1.0000 3250976.7529 0.4675 79 180   0.9244 0.8793 strategy_recall_first_r95
AHL_plain_ADPretrain S3 strategy_mild_stage_v2 ok 0.8880 0.8906 0.7215 3250976.7529 0.7972 57 7 173 22 0.9244 0.8793 mild_recall85_fpr10
AHL_plain_ADPretrain S3 strategy_stage_v3 ok 0.8764 0.7640 0.8608 3250976.7529 0.8095 68 21 159 11 0.9244 0.8793 recall_at_95_highest_threshold
AHL_plain_ADPretrain S3 recall_at_95_highest_threshold ok 0.8764 0.7640 0.8608 3250976.7529 0.8095 68 21 159 11 0.9244 0.8793 
AHL_plain_ADPretrain S3 recall_at_90_highest_threshold ok 0.8880 0.8906 0.7215 3250976.7529 0.7972 57 7 173 22 0.9244 0.8793 
AHL_plain_ADPretrain S3 strategy_balanced_f1 ok 0.8687 0.9245 0.6203 3250976.7529 0.7424 49 4 176 30 0.9244 0.8793 
AHL_plain_ADPretrain S4 production_normal_p95 ok 0.8571 0.9375 0.5696 3775593.1537 0.7087 45 3 177 34 0.8883 0.8444 
AHL_plain_ADPretrain S4 strategy_stage_adaptive ok 0.3282 0.3108 0.9873 3775593.1537 0.4727 78 173 7 1 0.8883 0.8444 strategy_recall_first_r95
AHL_plain_ADPretrain S4 strategy_mild_stage_v2 ok 0.8842 0.8657 0.7342 3775593.1537 0.7945 58 9 171 21 0.8883 0.8444 mild_recall85_fpr10
AHL_plain_ADPretrain S4 strategy_stage_v3 ok 0.7683 0.5798 0.8734 3775593.1537 0.6970 69 50 130 10 0.8883 0.8444 recall_at_95_highest_threshold
AHL_plain_ADPretrain S4 recall_at_95_highest_threshold ok 0.7683 0.5798 0.8734 3775593.1537 0.6970 69 50 130 10 0.8883 0.8444 
AHL_plain_ADPretrain S4 recall_at_90_highest_threshold ok 0.8842 0.8657 0.7342 3775593.1537 0.7945 58 9 171 21 0.8883 0.8444 
AHL_plain_ADPretrain S4 strategy_balanced_f1 ok 0.8649 0.8929 0.6329 3775593.1537 0.7407 50 6 174 29 0.8883 0.8444 
AHL_plain_ADPretrain S5 production_normal_p95 ok 0.8610 0.9057 0.6076 4233165.3461 0.7273 48 5 175 31 0.9372 0.8811 
AHL_plain_ADPretrain S5 strategy_stage_adaptive ok 0.3089 0.3062 1.0000 4233165.3461 0.4688 79 179 1  0.9372 0.8811 strategy_recall_first_r90
AHL_plain_ADPretrain S5 strategy_mild_stage_v2 ok 0.8958 0.8824 0.7595 4233165.3461 0.8163 60 8 172 19 0.9372 0.8811 mild_recall85_fpr10
AHL_plain_ADPretrain S5 strategy_stage_v3 ok 0.8842 0.8889 0.7089 4233165.3461 0.7887 56 7 173 23 0.9372 0.8811 recall_at_90_highest_threshold
AHL_plain_ADPretrain S5 recall_at_95_highest_threshold ok 0.8996 0.8533 0.8101 4233165.3461 0.8312 64 11 169 15 0.9372 0.8811 
AHL_plain_ADPretrain S5 recall_at_90_highest_threshold ok 0.8842 0.8889 0.7089 4233165.3461 0.7887 56 7 173 23 0.9372 0.8811 
AHL_plain_ADPretrain S5 strategy_balanced_f1 ok 0.8610 0.9057 0.6076 4233165.3461 0.7273 48 5 175 31 0.9372 0.8811 
AHL_plain_ADPretrain S6 production_normal_p95 ok 0.8880 0.9310 0.6835 4558703.0652 0.7883 54 4 176 25 0.9402 0.8922 
AHL_plain_ADPretrain S6 strategy_stage_adaptive ok 0.3089 0.3062 1.0000 4558703.0652 0.4688 79 179 1  0.9402 0.8922 strategy_recall_first_r90
AHL_plain_ADPretrain S6 strategy_mild_stage_v2 ok 0.9073 0.8873 0.7975 4558703.0652 0.8400 63 8 172 16 0.9402 0.8922 mild_recall85_fpr10
AHL_plain_ADPretrain S6 strategy_stage_v3 ok 0.9073 0.8873 0.7975 4558703.0652 0.8400 63 8 172 16 0.9402 0.8922 recall_at_90_highest_threshold
AHL_plain_ADPretrain S6 recall_at_95_highest_threshold ok 0.8880 0.7976 0.8481 4558703.0652 0.8221 67 17 163 12 0.9402 0.8922 
AHL_plain_ADPretrain S6 recall_at_90_highest_threshold ok 0.9073 0.8873 0.7975 4558703.0652 0.8400 63 8 172 16 0.9402 0.8922 
AHL_plain_ADPretrain S6 strategy_balanced_f1 ok 0.8649 0.9400 0.5949 4558703.0652 0.7287 47 3 177 32 0.9402 0.8922 
AHL_plain_ADPretrain S7 production_normal_p95 ok 0.8610 0.9216 0.5949 4670110.5652 0.7231 47 4 176 32 0.9373 0.8807 
AHL_plain_ADPretrain S7 strategy_stage_adaptive ok 0.8610 0.9057 0.6076 4670110.5652 0.7273 48 5 175 31 0.9373 0.8807 strategy_balanced_f1
AHL_plain_ADPretrain S7 strategy_mild_stage_v2 ok 0.8610 0.9057 0.6076 4670110.5652 0.7273 48 5 175 31 0.9373 0.8807 balanced_f1_late
AHL_plain_ADPretrain S7 strategy_stage_v3 ok 0.8610 0.9057 0.6076 4670110.5652 0.7273 48 5 175 31 0.9373 0.8807 strategy_balanced_f1
AHL_plain_ADPretrain S7 recall_at_95_highest_threshold ok 0.8880 0.7907 0.8608 4670110.5652 0.8242 68 18 162 11 0.9373 0.8807 
AHL_plain_ADPretrain S7 recall_at_90_highest_threshold ok 0.8880 0.8676 0.7468 4670110.5652 0.8027 59 9 171 20 0.9373 0.8807 
AHL_plain_ADPretrain S7 strategy_balanced_f1 ok 0.8610 0.9057 0.6076 4670110.5652 0.7273 48 5 175 31 0.9373 0.8807 
AHL_plain_ADPretrain S8 production_normal_p95 ok 0.8687 0.8947 0.6456 4717836.0766 0.7500 51 6 174 28 0.9212 0.8782 
AHL_plain_ADPretrain S8 strategy_stage_adaptive ok 0.8687 0.9592 0.5949 4717836.0766 0.7344 47 2 178 32 0.9212 0.8782 strategy_balanced_f1
AHL_plain_ADPretrain S8 strategy_mild_stage_v2 ok 0.8687 0.9592 0.5949 4717836.0766 0.7344 47 2 178 32 0.9212 0.8782 balanced_f1_late
AHL_plain_ADPretrain S8 strategy_stage_v3 ok 0.8687 0.9592 0.5949 4717836.0766 0.7344 47 2 178 32 0.9212 0.8782 strategy_balanced_f1
AHL_plain_ADPretrain S8 recall_at_95_highest_threshold ok 0.8610 0.7263 0.8734 4717836.0766 0.7931 69 26 154 10 0.9212 0.8782 
AHL_plain_ADPretrain S8 recall_at_90_highest_threshold ok 0.8958 0.8939 0.7468 4717836.0766 0.8138 59 7 173 20 0.9212 0.8782 
AHL_plain_ADPretrain S8 strategy_balanced_f1 ok 0.8687 0.9592 0.5949 4717836.0766 0.7344 47 2 178 32 0.9212 0.8782 
YOLO26s_cls S0 ALL missing            
YOLO26s_cls S1 production_normal_p95 ok 0.7992 0.9091 0.3797 86.1174 0.5357 30 3 177 49 0.8534 0.7917 
YOLO26s_cls S1 strategy_stage_adaptive ok 0.3089 0.3062 1.0000 86.1174 0.4688 79 179 1  0.8534 0.7917 strategy_recall_first_r95
YOLO26s_cls S1 strategy_mild_stage_v2 ok 0.4363 0.3484 0.9747 86.1174 0.5133 77 144 36 2 0.8534 0.7917 mild_recall90_fpr20
YOLO26s_cls S1 strategy_stage_v3 ok 0.5058 0.3793 0.9747 86.1174 0.5461 77 126 54 2 0.8534 0.7917 recall_at_95_highest_threshold
YOLO26s_cls S1 recall_at_95_highest_threshold ok 0.5058 0.3793 0.9747 86.1174 0.5461 77 126 54 2 0.8534 0.7917 
YOLO26s_cls S1 recall_at_90_highest_threshold ok 0.5058 0.3793 0.9747 86.1174 0.5461 77 126 54 2 0.8534 0.7917 
YOLO26s_cls S1 strategy_balanced_f1 ok 0.8301 0.7692 0.6329 86.1174 0.6944 50 15 165 29 0.8534 0.7917 
YOLO26s_cls S2 production_normal_p95 ok 0.8340 0.9091 0.5063 52.7447 0.6504 40 4 176 39 0.8565 0.8024 
YOLO26s_cls S2 strategy_stage_adaptive ok 0.3243 0.3110 1.0000 52.7447 0.4745 79 175 5  0.8565 0.8024 strategy_recall_first_r95
YOLO26s_cls S2 strategy_mild_stage_v2 ok 0.6641 0.4733 0.8987 52.7447 0.6201 71 79 101 8 0.8565 0.8024 mild_recall90_fpr20
YOLO26s_cls S2 strategy_stage_v3 ok 0.6641 0.4733 0.8987 52.7447 0.6201 71 79 101 8 0.8565 0.8024 recall_at_95_highest_threshold
YOLO26s_cls S2 recall_at_95_highest_threshold ok 0.6641 0.4733 0.8987 52.7447 0.6201 71 79 101 8 0.8565 0.8024 
YOLO26s_cls S2 recall_at_90_highest_threshold ok 0.6680 0.4748 0.8354 52.7447 0.6055 66 73 107 13 0.8565 0.8024 
YOLO26s_cls S2 strategy_balanced_f1 ok 0.8417 0.9750 0.4937 52.7447 0.6555 39 1 179 40 0.8565 0.8024 
YOLO26s_cls S3 production_normal_p95 ok 0.8533 0.9556 0.5443 28.0535 0.6935 43 2 178 36 0.8645 0.8492 
YOLO26s_cls S3 strategy_stage_adaptive ok 0.3050 0.3050 1.0000 28.0535 0.4675 79 180   0.8645 0.8492 strategy_recall_first_r95
YOLO26s_cls S3 strategy_mild_stage_v2 ok 0.4015 0.3319 0.9494 28.0535 0.4918 75 151 29 4 0.8645 0.8492 mild_recall85_fpr10
YOLO26s_cls S3 strategy_stage_v3 ok 0.4131 0.3363 0.9494 28.0535 0.4967 75 148 32 4 0.8645 0.8492 recall_at_95_highest_threshold
YOLO26s_cls S3 recall_at_95_highest_threshold ok 0.4131 0.3363 0.9494 28.0535 0.4967 75 148 32 4 0.8645 0.8492 
YOLO26s_cls S3 recall_at_90_highest_threshold ok 0.4402 0.3472 0.9494 28.0535 0.5085 75 141 39 4 0.8645 0.8492 
YOLO26s_cls S3 strategy_balanced_f1 ok 0.8263 1.0000 0.4304 28.0535 0.6018 34  180 45 0.8645 0.8492 
YOLO26s_cls S4 production_normal_p95 ok 0.8842 0.9804 0.6329 24.2835 0.7692 50 1 179 29 0.9003 0.8750 
YOLO26s_cls S4 strategy_stage_adaptive ok 0.3089 0.3062 1.0000 24.2835 0.4688 79 179 1  0.9003 0.8750 strategy_recall_first_r95
YOLO26s_cls S4 strategy_mild_stage_v2 ok 0.8958 0.9483 0.6962 24.2835 0.8029 55 3 177 24 0.9003 0.8750 mild_recall85_fpr10
YOLO26s_cls S4 strategy_stage_v3 ok 0.8764 0.8615 0.7089 24.2835 0.7778 56 9 171 23 0.9003 0.8750 recall_at_95_highest_threshold
YOLO26s_cls S4 recall_at_95_highest_threshold ok 0.8764 0.8615 0.7089 24.2835 0.7778 56 9 171 23 0.9003 0.8750 
YOLO26s_cls S4 recall_at_90_highest_threshold ok 0.8958 0.9483 0.6962 24.2835 0.8029 55 3 177 24 0.9003 0.8750 
YOLO26s_cls S4 strategy_balanced_f1 ok 0.8958 0.9483 0.6962 24.2835 0.8029 55 3 177 24 0.9003 0.8750 
YOLO26s_cls S5 production_normal_p95 ok 0.8919 0.8923 0.7342 26.9950 0.8056 58 7 173 21 0.8978 0.8836 
YOLO26s_cls S5 strategy_stage_adaptive ok 0.3127 0.3059 0.9873 26.9950 0.4671 78 177 3 1 0.8978 0.8836 strategy_recall_first_r90
YOLO26s_cls S5 strategy_mild_stage_v2 ok 0.8533 0.7412 0.7975 26.9950 0.7683 63 22 158 16 0.8978 0.8836 mild_recall85_fpr10
YOLO26s_cls S5 strategy_stage_v3 ok 0.8610 0.7590 0.7975 26.9950 0.7778 63 20 160 16 0.8978 0.8836 recall_at_90_highest_threshold
YOLO26s_cls S5 recall_at_95_highest_threshold ok 0.7143 0.5182 0.8987 26.9950 0.6574 71 66 114 8 0.8978 0.8836 
YOLO26s_cls S5 recall_at_90_highest_threshold ok 0.8610 0.7590 0.7975 26.9950 0.7778 63 20 160 16 0.8978 0.8836 
YOLO26s_cls S5 strategy_balanced_f1 ok 0.9073 0.9825 0.7089 26.9950 0.8235 56 1 179 23 0.8978 0.8836 
YOLO26s_cls S6 production_normal_p95 ok 0.9266 0.9412 0.8101 24.4326 0.8707 64 4 176 15 0.9733 0.9592 
YOLO26s_cls S6 strategy_stage_adaptive ok 0.3282 0.3123 1.0000 24.4326 0.4759 79 174 6  0.9733 0.9592 strategy_recall_first_r90
YOLO26s_cls S6 strategy_mild_stage_v2 ok 0.9228 0.9538 0.7848 24.4326 0.8611 62 3 177 17 0.9733 0.9592 mild_recall85_fpr10
YOLO26s_cls S6 strategy_stage_v3 ok 0.9228 0.9538 0.7848 24.4326 0.8611 62 3 177 17 0.9733 0.9592 recall_at_90_highest_threshold
YOLO26s_cls S6 recall_at_95_highest_threshold ok 0.9421 0.8810 0.9367 24.4326 0.9080 74 10 170 5 0.9733 0.9592 
YOLO26s_cls S6 recall_at_90_highest_threshold ok 0.9228 0.9538 0.7848 24.4326 0.8611 62 3 177 17 0.9733 0.9592 
YOLO26s_cls S6 strategy_balanced_f1 ok 0.9228 0.9538 0.7848 24.4326 0.8611 62 3 177 17 0.9733 0.9592 
YOLO26s_cls S7 production_normal_p95 ok 0.9189 0.8919 0.8354 23.9924 0.8627 66 8 172 13 0.9589 0.9370 
YOLO26s_cls S7 strategy_stage_adaptive ok 0.9189 0.9677 0.7595 23.9924 0.8511 60 2 178 19 0.9589 0.9370 strategy_balanced_f1
YOLO26s_cls S7 strategy_mild_stage_v2 ok 0.9189 0.9677 0.7595 23.9924 0.8511 60 2 178 19 0.9589 0.9370 balanced_f1_late
YOLO26s_cls S7 strategy_stage_v3 ok 0.9189 0.9677 0.7595 23.9924 0.8511 60 2 178 19 0.9589 0.9370 strategy_balanced_f1
YOLO26s_cls S7 recall_at_95_highest_threshold ok 0.8842 0.7753 0.8734 23.9924 0.8214 69 20 160 10 0.9589 0.9370 
YOLO26s_cls S7 recall_at_90_highest_threshold ok 0.9189 0.9394 0.7848 23.9924 0.8552 62 4 176 17 0.9589 0.9370 
YOLO26s_cls S7 strategy_balanced_f1 ok 0.9189 0.9677 0.7595 23.9924 0.8511 60 2 178 19 0.9589 0.9370 
YOLO26s_cls S8 production_normal_p95 ok 0.9228 0.9041 0.8354 24.7711 0.8684 66 7 173 13 0.9677 0.9488 
YOLO26s_cls S8 strategy_stage_adaptive ok 0.9151 0.9254 0.7848 24.7711 0.8493 62 5 175 17 0.9677 0.9488 strategy_balanced_f1
YOLO26s_cls S8 strategy_mild_stage_v2 ok 0.9151 0.9254 0.7848 24.7711 0.8493 62 5 175 17 0.9677 0.9488 balanced_f1_late
YOLO26s_cls S8 strategy_stage_v3 ok 0.9151 0.9254 0.7848 24.7711 0.8493 62 5 175 17 0.9677 0.9488 strategy_balanced_f1
YOLO26s_cls S8 recall_at_95_highest_threshold ok 0.9189 0.9028 0.8228 24.7711 0.8609 65 7 173 14 0.9677 0.9488 
YOLO26s_cls S8 recall_at_90_highest_threshold ok 0.9189 0.9531 0.7722 24.7711 0.8531 61 3 177 18 0.9677 0.9488 
YOLO26s_cls S8 strategy_balanced_f1 ok 0.9151 0.9254 0.7848 24.7711 0.8493 62 5 175 17 0.9677 0.9488 
