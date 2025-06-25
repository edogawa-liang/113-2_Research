#!/bin/bash
set -e
set -x 


# ==================== CF 生成解釋 ========================

## Feature to Node (只有特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type CFExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn
python stage2_expsubg.py --dataset CiteSeer --explainer_type CFExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn 
python stage2_expsubg.py --dataset Amazon --explainer_type CFExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn 
python stage2_expsubg.py --dataset FacebookPagePage --explainer_type CFExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn 
python stage2_expsubg.py --dataset GitHub --explainer_type CFExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn 

# 
wait