#!/bin/bash
set -e
set -x 

# ==================== CF 生成解釋 ========================
# 可以跑 但feature to node 乍看學不太起來

## 原始 Graph (只有節點邊)
python stage2_expsubg.py --dataset Cora --explainer_type CFExplainer --run_mode stage2 --stage1_path saved/stage1
python stage2_expsubg.py --dataset CiteSeer --explainer_type CFExplainer --run_mode stage2 --stage1_path saved/stage1
python stage2_expsubg.py --dataset Amazon --explainer_type CFExplainer --run_mode stage2 --stage1_path saved/stage1
python stage2_expsubg.py --dataset FacebookPagePage  --normalize --explainer_type CFExplainer --run_mode stage2 --stage1_path saved/stage1
python stage2_expsubg.py --dataset GitHub  --normalize --explainer_type CFExplainer --run_mode stage2 --stage1_path saved/stage1

## Feature to Node (含節點邊 & 特徵邊)
## Feature to Node (只有特徵邊)

wait