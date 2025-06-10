#!/bin/bash
set -e
set -x 

# 計算重要度
# 1. 原圖 計算 [Degree, PageRank, Betweenness, Closeness] 
# python tools/cal_node_imp.py --dataset Cora 
# python tools/cal_node_imp.py --dataset CiteSeer
# python tools/cal_node_imp.py --dataset Amazon
# python tools/cal_node_imp.py --dataset FacebookPagePage
# python tools/cal_node_imp.py --dataset GitHub

# # 2. feature to node (包含 node-node, node-feature)
# python tools/cal_node_imp.py --dataset Cora --feature_to_node
# python tools/cal_node_imp.py --dataset CiteSeer --feature_to_node
# python tools/cal_node_imp.py --dataset Amazon --feature_to_node 
# python tools/cal_node_imp.py --dataset FacebookPagePage --feature_to_node
# python tools/cal_node_imp.py --dataset GitHub --feature_to_node

# # 3. feature to node (只有 node-feature)
# python tools/cal_node_imp.py --dataset Cora --feature_to_node --only_feature_node
# python tools/cal_node_imp.py --dataset CiteSeer --feature_to_node --only_feature_node
# python tools/cal_node_imp.py --dataset Amazon --feature_to_node --only_feature_node
# python tools/cal_node_imp.py --dataset FacebookPagePage --feature_to_node --only_feature_node
# python tools/cal_node_imp.py --dataset GitHub --feature_to_node --only_feature_node


# 切分資料集
python data/prepare_split.py --dataset Cora --num_repeat 10 --train_ratio 0.2 --val_ratio 0.05 --test_ratio 0.1
python data/prepare_split.py --dataset CiteSeer --num_repeat 10 --train_ratio 0.2 --val_ratio 0.05 --test_ratio 0.1


# Stage 1: 訓練原模型
# 1. 原圖 (連續型記得+normalize)
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original --note original
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original --note original


# 2. 原圖 (只有結構)
## (1) random32+imp
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure --note random+imp --only_structure --structure_mode random+imp --learn_embedding
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure --note random+imp --only_structure --structure_mode random+imp --learn_embedding


# ## (2) one+imp
# python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_one --note one+imp --only_structure --structure_mode one+imp
# python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_one --note one+imp --only_structure --structure_mode one+imp


# 3. Feature to Node (含節點邊 & 特徵邊)
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn --note feature2node_nn_fn  --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding

# 4. Feature to Node (只有特徵邊)
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn --note feature2node_fn  --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding

# Stage 2: 生成解釋子圖 (GNNExplainer)

# 1. 原圖
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original 

# 2. 原圖 (只有結構)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp 

# 3. Feature to Node (含節點邊 & 特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp 

# 4. Feature to Node (只有特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 

wait