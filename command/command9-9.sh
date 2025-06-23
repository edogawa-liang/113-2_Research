#!/bin/bash
set -e
set -x 


# =================== 重新生成經過 feature to node 的解釋子圖 =========================
# 注意不要先normalize
# 1. train base model
# ------------------------------------------------------------
## Feature to Node (只有特徵邊)
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn  --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset FacebookPagePage --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset GitHub --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding


# 2. 生成解釋
## Feature to Node (只有特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset FacebookPagePage--explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset GitHub --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 


# 
wait