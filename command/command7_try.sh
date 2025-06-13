#!/bin/bash
set -e
set -x 

# =============== try =============== 
# 切分資料集
python data/prepare_split.py --dataset Cora --num_repeat 10 --train_ratio 0.2 --val_ratio 0.05 --test_ratio 0.1
python data/prepare_split.py --dataset FacebookPagePage --num_repeat 10 --train_ratio 0.2 --val_ratio 0.05 --test_ratio 0.1

# Stage 1: 訓練原模型
# 1. 原圖 (連續型記得+normalize)
python training_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original --note original
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original --note original


# 2. 原圖 (只有結構)
## (1) random32+imp
python training_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_structure --note random+imp --only_structure --structure_mode random+imp --learn_embedding
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_structure --note random+imp --only_structure --structure_mode random+imp --learn_embedding


## (2) one+imp
python training_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_structure_one --note one+imp --only_structure --structure_mode one+imp
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_structure_one --note one+imp --only_structure --structure_mode one+imp


# 3. Feature to Node (含節點邊 & 特徵邊)
python training_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_feature2node_nn_fn --note feature2node_nn_fn  --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding

# 4. Feature to Node (只有特徵邊)
python training_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_feature2node_fn --note feature2node_fn  --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding


# Stage 2: 生成解釋子圖 (GNNExplainer)

# 1. 原圖
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode try_stage2_original_structure --stage1_path saved/try_original --epoch 1
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode try_stage2_original_structure --stage1_path saved/try_original  --epoch 1

# 2. 原圖 (只有結構)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode try_stage2_original_structure --stage1_path saved/try_original_structure --only_structure --structure_mode random+imp --epoch 1
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode try_stage2_original_structure --stage1_path saved/try_original_structure --only_structure --structure_mode random+imp --epoch 1

# 3. Feature to Node (含節點邊 & 特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode try_stage2_original_feature2node_nn_fn --stage1_path saved/try_original_feature2node_nn_fn --feature_to_node --structure_mode random+imp --epoch 1 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode try_stage2_original_feature2node_nn_fn --stage1_path saved/try_original_feature2node_nn_fn --feature_to_node --structure_mode random+imp --epoch 1

# 4. Feature to Node (只有特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode try_stage2_original_feature2node_fn --stage1_path saved/try_original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --epoch 1 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode try_stage2_original_feature2node_fn --stage1_path saved/try_original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --epoch 1 


# 切分測試集
python data/split_unknown_to_test.py --dataset Cora --use_id 0 --num_repeat 10

# Stage 1 (part2): 訓練原模型 (train/val 固定 (以 split0 為例)，test 重抽，抽10次)

python training_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original --note original_split0 --fix_train_valid --split_start 0 --split_end 0

# ======================

# Stage 3: 訓練移除解釋子圖後模型
# 1. 移除結構
## (1) Original Graph (edge mask)
### [1] random select
### [2] explainer select
### [3] random walk select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/try_stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode try_remove_from_GNNExplainer --note split0_original_only_structure
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 1 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/try_stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode try_remove_from_GNNExplainer

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --selector_type random --fraction 0.1 --run_mode try_baseline_Result
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --selector_type random_walk --walk_length 5 --num_walks 3 --fraction 0.1 --run_mode try_remove_from_RandomWalk


## (2) Only Structure
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/try_stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode try_remove_from_GNNExplainer --note split0_original_only_structure --only_structure
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 1 --lr 0.01 --selector_type random_walk --walk_length 10 --num_walks 5 --fraction 0.1 --node_choose all_train --run_mode try_remove_from_RandomWalk --only_structure


# 2. 移除特徵
## (1) Original Graph (node mask)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --selector_type random_walk --walk_length 5 --num_walks 3 --fraction 0.1 --run_mode try_remove_from_RandomWalk --fraction_feat 0.1 --node_choose stratified_by_degree

## (2) feature to Node (含節點邊 & 特徵邊)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --selector_type random_walk --walk_length 1 --num_walks 1 --run_mode try_remove_from_RandomWalk --feature_to_node --structure_mode random+imp --fraction_feat 0.1 --node_choose stratified_by_degree --only_feature_node

## (3) feature to Node (只有特徵邊)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --selector_type random_walk --walk_length 1 --num_walks 1 --fraction 0.1 --run_mode try_remove_from_RandomWalk --feature_to_node --structure_mode random+imp --fraction_feat 0.1 --same_feat --node_choose stratified_by_degree


# 3. 移除結構+特徵 (特徵相同)
## (1) Original Graph
## (2) feature to Node (含節點邊 & 特徵邊)
## (3) Only Structure + Feature to Node (只有特徵邊) 

# 4. 移除結構+特徵 (特徵不同)
## (1) Original Graph
## (2) feature to Node (含節點邊 & 特徵邊)
## (3) Only Structure + Feature to Node (只有特徵邊)



# 表格式資料
wait