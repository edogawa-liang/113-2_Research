#!/bin/bash
set -e
set -x 

# 還沒跑 (split1-9的解釋子圖)
# Stage 2: 生成解釋子圖 (GNNExplainer)
# 1. 原圖
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original --repeat_start 1 --repeat_end 9
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original --repeat_start 1 --repeat_end 9
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original --repeat_start 1 --repeat_end 9
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original --repeat_start 1 --repeat_end 9
python stage2_expsubg.py --dataset GitHub --normalize --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original --repeat_start 1 --repeat_end 9

# 2. 原圖 (只有結構)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp --repeat_start 1 --repeat_end 9
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp --repeat_start 1 --repeat_end 9
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp --repeat_start 1 --repeat_end 9 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp --repeat_start 1 --repeat_end 9 
python stage2_expsubg.py --dataset GitHub --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp --repeat_start 1 --repeat_end 9 

# 3. Feature to Node (含節點邊 & 特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp --repeat_start 1 --repeat_end 9 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp --repeat_start 1 --repeat_end 9 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp --repeat_start 1 --repeat_end 9 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp --repeat_start 1 --repeat_end 9 
python stage2_expsubg.py --dataset GitHub --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp --repeat_start 1 --repeat_end 9 

# 4. Feature to Node (只有特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --repeat_start 1 --repeat_end 9 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --repeat_start 1 --repeat_end 9 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --repeat_start 1 --repeat_end 9 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --repeat_start 1 --repeat_end 9 
python stage2_expsubg.py --dataset GitHub --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --repeat_start 1 --repeat_end 9 

wait