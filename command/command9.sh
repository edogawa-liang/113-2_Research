#!/bin/bash
set -e
set -x 

# 把 stage2 和 stage2_structure 從 saved_0621 直接移過來 (command 7 已跑過) 
# 但之後不小心誤刪 saved_0621.....


# ============================================
# 誤刪了: core_subgraph.... 重跑 GNNExplainer 和 Random 移除邊的實驗
# command8-1.sh


# ============================================
# 天花板 (只使用結構的GNN (移除所有特徵), MLP (Graph移除所有邊))

# 1. train GNN only look at structure
# ------------------------------------------------------------
# command7 中跑過了

# 2. train MLP with no feature
# ------------------------------------------------------------
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp --fix_train_valid --split_id 0
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp --fix_train_valid --split_id 0
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp --fix_train_valid --split_id 0
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp --fix_train_valid --split_id 0
python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure_split0 --note random+imp_split0 --only_structure --structure_mode random+imp --fix_train_valid --split_id 0


# ============================================

# + 重跑 10 次
# ============================================
# 
wait