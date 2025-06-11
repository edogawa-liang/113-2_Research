#!/bin/bash
set -e
set -x 

# # 切分測試集 (都使用第0組dataset去切分)
# python data/split_unknown_to_test.py --dataset Cora --use_id 0 --num_repeat 10
# python data/split_unknown_to_test.py --dataset CiteSeer --use_id 0 --num_repeat 10
# python data/split_unknown_to_test.py --dataset Amazon --use_id 0 --num_repeat 10
# python data/split_unknown_to_test.py --dataset FacebookPagePage --use_id 0 --num_repeat 10
# python data/split_unknown_to_test.py --dataset GitHub --use_id 0 --num_repeat 10


# Stage 1 (part2): 訓練原模型 (train/val 固定 (以 split0 為例)，test 重抽，抽10次)
# 1. 原圖 (連續型記得+normalize)
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_split0 --note original_split0 --fix_train_valid --split_id 0
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_split0 --note original_split0 --fix_train_valid --split_id 0
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_split0 --note original_split0 --fix_train_valid --split_id 0
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_split0 --note original_split0 --fix_train_valid --split_id 0
python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_split0 --note original_split0 --fix_train_valid --split_id 0


# 2. 原圖 (只有結構)
## (1) random32+imp
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0
python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0


# 3. Feature to Node (含節點邊 & 特徵邊)
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn_split0 --note feature2node_nn_fn_split0  --feature_to_node --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn_split0 --note feature2node_nn_fn_split0 --feature_to_node --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn_split0 --note feature2node_nn_fn_split0 --feature_to_node --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn_split0 --note feature2node_nn_fn_split0 --feature_to_node --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0
python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_nn_fn_split0 --note feature2node_nn_fn_split0 --feature_to_node --structure_mode random+imp --learn_embedding --fix_train_valid --split_id 0

# 4. Feature to Node (只有特徵邊)
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn_split0 --note feature2node_fn_split0  --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding --fix_train_valid --split_id 0
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn_split0 --note feature2node_fn_split0 --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding --fix_train_valid --split_id 0
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn_split0 --note feature2node_fn_split0 --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding --fix_train_valid --split_id 0
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn_split0 --note feature2node_fn_split0 --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding --fix_train_valid --split_id 0
python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node_fn_split0 --note feature2node_fn_split0 --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding --fix_train_valid --split_id 0




wait