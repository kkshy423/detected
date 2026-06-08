# 20260602 Early Conservative Bridge Decision

Decision rule: if the selected early bridge candidate improves F1 by at least 0.01 over ADP-only, or reduces FP while FN does not increase by more than 2, or improves AUC-PR by at least 0.01, and the stage recall floor is satisfied, then the stage can adopt early bridge.

## S1

- Selected alpha: `0.7000`
- ADP-only: F1=0.7753, FP=39, FN=1, AUC-PR=0.9151
- AHL-only: F1=0.7159, FP=43, FN=7, AUC-PR=0.8926
- Early bridge: F1=0.8092, FP=33, FN=0, AUC-PR=0.9324
- Recall floor satisfied: `True`
- Decision: **introduce early bridge**

## S2

- Selected alpha: `0.7000`
- ADP-only: F1=0.7753, FP=39, FN=1, AUC-PR=0.9151
- AHL-only: F1=0.5317, FP=115, FN=3, AUC-PR=0.8122
- Early bridge: F1=0.8000, FP=35, FN=0, AUC-PR=0.9318
- Recall floor satisfied: `True`
- Decision: **introduce early bridge**

## S3

- Selected alpha: `0.6000`
- ADP-only: F1=0.7931, FP=35, FN=1, AUC-PR=0.9151
- AHL-only: F1=0.7665, FP=33, FN=6, AUC-PR=0.8905
- Early bridge: F1=0.8187, FP=31, FN=0, AUC-PR=0.9360
- Recall floor satisfied: `True`
- Decision: **introduce early bridge**

## S4

- Selected alpha: `0.6000`
- ADP-only: F1=0.7931, FP=35, FN=1, AUC-PR=0.9151
- AHL-only: F1=0.7602, FP=36, FN=5, AUC-PR=0.9041
- Early bridge: F1=0.9139, FP=12, FN=1, AUC-PR=0.9437
- Recall floor satisfied: `True`
- Decision: **introduce early bridge**
