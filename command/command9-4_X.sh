#!/bin/bash
set -e
set -x 

# 2. 生成解釋
# ------------------------------------------------------------
## Feature to Node (只有特徵邊)
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset GitHub --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 

# ============================================
# 
wait