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
python data/prepare_split.py --dataset Amazon --num_repeat 10 --train_ratio 0.2 --val_ratio 0.05 --test_ratio 0.1
python data/prepare_split.py --dataset FacebookPagePage --num_repeat 10 --train_ratio 0.2 --val_ratio 0.05 --test_ratio 0.1
python data/prepare_split.py --dataset GitHub --num_repeat 10 --train_ratio 0.2 --val_ratio 0.05 --test_ratio 0.1


# Stage 1: 訓練原模型 (train/val/test 整組重抽，抽10次)
# 1. 原圖 (連續型記得+normalize)
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original --note original
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original --note original
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original --note original
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original --note original
python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original --note original


# 2. 原圖 (只有結構)
## (1) random32+imp
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure --note random+imp --only_structure --structure_mode random+imp --learn_embedding
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure --note random+imp --only_structure --structure_mode random+imp --learn_embedding
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure --note random+imp --only_structure --structure_mode random+imp --learn_embedding
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure --note random+imp --only_structure --structure_mode random+imp --learn_embedding
python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure --note random+imp --only_structure --structure_mode random+imp --learn_embedding


# ## (2) one+imp
# python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_one --note one+imp --only_structure --structure_mode one+imp
# python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_one --note one+imp --only_structure --structure_mode one+imp
# python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_one --note one+imp --only_structure --structure_mode one+imp
# python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_one --note one+imp --only_structure --structure_mode one+imp
# python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_one --note one+imp --only_structure --structure_mode one+imp


# 3. Feature to Node (含節點邊 & 特徵邊)
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn --note feature2node_nn_fn  --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding

# 4. Feature to Node (只有特徵邊)
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn --note feature2node_fn  --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding


# Stage 2: 生成解釋子圖 (GNNExplainer)
# 暫時只有跑 split 0 的資料集
# 1. 原圖
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original 
python stage2_expsubg.py --dataset GitHub --normalize --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original 

# 2. 原圖 (只有結構)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp 
python stage2_expsubg.py --dataset GitHub --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp 

# 3. Feature to Node (含節點邊 & 特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
python stage2_expsubg.py --dataset GitHub --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/original_feature2node_nn_fn --feature_to_node --structure_mode random+imp 

# 4. Feature to Node (只有特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset GitHub --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 

# ========

# 切分測試集 (都使用第0組dataset去切分)
python data/split_unknown_to_test.py --dataset Cora --use_id 0 --num_repeat 10
python data/split_unknown_to_test.py --dataset CiteSeer --use_id 0 --num_repeat 10
python data/split_unknown_to_test.py --dataset Amazon --use_id 0 --num_repeat 10
python data/split_unknown_to_test.py --dataset FacebookPagePage --use_id 0 --num_repeat 10
python data/split_unknown_to_test.py --dataset GitHub --use_id 0 --num_repeat 10


# Stage 1 (part2): 訓練原模型 (train/val 固定 (以 split0 為例)，test 重抽，抽10次)
# 1. 原圖 (連續型記得+normalize)
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_split0 --note original_split0 --fix_train_valid --split_id 0
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_split0 --note original_split0 --fix_train_valid --split_id 0
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_split0 --note original_split0 --fix_train_valid --split_id 0
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_split0 --note original_split0 --fix_train_valid --split_id 0
python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_split0 --note original_split0 --fix_train_valid --split_id 0

# 重抽test的目的是檢查移除核心子圖後有沒有大掉，而比較對象只有original，經過feature to node 最後也會轉成ori graph，所以feature to node 的結果不需要重抽test

# # 2. 原圖 (只有結構)
# ## (1) random32+imp
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp --fix_train_valid --split_id 0
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp--fix_train_valid --split_id 0
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp --fix_train_valid --split_id 0
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp --fix_train_valid --split_id 0
python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp --fix_train_valid --split_id 0


# # 3. Feature to Node (含節點邊 & 特徵邊)
# python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn_split0 --note feature2node_nn_fn_split0  --feature_to_node --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0
# python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn_split0 --note feature2node_nn_fn_split0 --feature_to_node --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0
# python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn_split0 --note feature2node_nn_fn_split0 --feature_to_node --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0
# python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn_split0 --note feature2node_nn_fn_split0 --feature_to_node --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0
# python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn_split0 --note feature2node_nn_fn_split0 --feature_to_node --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0

# # 4. Feature to Node (只有特徵邊)
# python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn_split0 --note feature2node_fn_split0  --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding --fix_train_valid --split_id 0
# python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn_split0 --note feature2node_fn_split0 --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding --fix_train_valid --split_id 0
# python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn_split0 --note feature2node_fn_split0 --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding --fix_train_valid --split_id 0
# python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn_split0 --note feature2node_fn_split0 --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding --fix_train_valid --split_id 0
# python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn_split0 --note feature2node_fn_split0 --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding --fix_train_valid --split_id 0


# Stage 3 移除核心子圖


# 
wait