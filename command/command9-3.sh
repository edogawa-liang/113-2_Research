#!/bin/bash
set -e
set -x 

# 2. 生成解釋
# ------------------------------------------------------------
## Feature to Node (含節點邊 & 特徵邊)
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
python stage2_expsubg.py --dataset GitHub --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
# ============================================
# 
wait