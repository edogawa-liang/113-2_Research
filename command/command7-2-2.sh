#!/bin/bash
set -e
set -x 

# Stage 2: 生成解釋子圖 (GNNExplainer)
# 3. Feature to Node (含節點邊 & 特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp --repeat_start 0 --repeat_end 0 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp --repeat_start 0 --repeat_end 0 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp --repeat_start 0 --repeat_end 0 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp --repeat_start 0 --repeat_end 0 
python stage2_expsubg.py --dataset GitHub --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp --repeat_start 0 --repeat_end 0 

# 4. Feature to Node (只有特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --repeat_start 0 --repeat_end 0 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --repeat_start 0 --repeat_end 0 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --repeat_start 0 --repeat_end 0 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --repeat_start 0 --repeat_end 0 
python stage2_expsubg.py --dataset GitHub --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --repeat_start 0 --repeat_end 0 

wait