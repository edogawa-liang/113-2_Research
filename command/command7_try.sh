#!/bin/bash
set -e
set -x 

# =============== try =============== 
# # 切分資料集
# python data/prepare_split.py --dataset Cora --num_repeat 10 --train_ratio 0.2 --val_ratio 0.05 --test_ratio 0.1

# # Stage 1: 訓練原模型
# # 1. 原圖 (連續型記得+normalize)
# python training_main.py --dataset Cora --model GCN2 --epochs 2 --lr 0.01 --run_mode try_original --note original


# 2. 原圖 (只有結構)
## (1) random32+imp
# python training_main.py --dataset Cora --model GCN2 --epochs 2 --lr 0.01 --run_mode try_original_structure --note random+imp --only_structure --structure_mode random+imp


# ## (2) one+imp
# python training_main.py --dataset Cora --model GCN2 --epochs 2 --lr 0.01 --run_mode try_original_structure_one --note one+imp --only_structure --structure_mode one+imp


# # 3. Feature to Node (含節點邊 & 特徵邊)
# python training_main.py --dataset Cora --model GCN2 --epochs 2 --lr 0.01 --run_mode try_original_feature2node_nn_fn --note feature2node_nn_fn  --feature_to_node --structure_mode random+imp

# 4. Feature to Node (只有特徵邊)
python training_main.py --dataset Cora --model GCN2 --epochs 2 --lr 0.01 --run_mode try_original_feature2node_fn --note feature2node_fn  --feature_to_node --structure_mode random+imp --only_feature_node


# Stage 2: 生成解釋子圖
# 1. 原圖
# 2. 原圖 (只有結構)
# 3. Feature to Node (含節點邊 & 特徵邊)
# 4. Feature to Node (只有特徵邊)

wait