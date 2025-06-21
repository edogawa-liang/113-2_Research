#!/bin/bash
set -e
set -x 

# 把 stage2 和 stage2_structure 從 saved_0621 直接移過來 (command 7 已跑過) 
# 但之後不小心誤刪 saved_0621.....


# ============================================
# 誤刪了: core_subgraph.... 重跑 GNNExplainer 和 Random 移除邊的實驗

# command8-1.sh


# ====================天花板 (只使用結構的GNN (移除所有特徵), MLP (Graph移除所有邊))========================

# 1. train GNN only look at structure
# ------------------------------------------------------------
# command7 中跑過了

# 2. train MLP (only look at features)
# ------------------------------------------------------------
python training_main.py --dataset Cora --model MLP --epochs 1000 --lr 0.01 --run_mode stage1_mlp  --split_start 0 --split_end 0 
python training_main.py --dataset CiteSeer --model MLP --epochs 1000 --lr 0.01 --run_mode stage1_mlp --split_start 0 --split_end 0 
python training_main.py --dataset Amazon --model MLP --epochs 1000 --lr 0.01 --run_mode stage1_mlp --split_start 0 --split_end 0 
python training_main.py --dataset FacebookPagePage --normalize --model MLP --epochs 1000 --lr 0.01 --run_mode stage1_mlp --split_start 0 --split_end 0 
python training_main.py --dataset GitHub --normalize --model MLP --epochs 1000 --lr 0.01 --run_mode stage1_mlp --split_start 0 --split_end 0 


# 3. Test 10 times
# ------------------------------------------------------------
## Original GNN
python testing_main.py --dataset Cora --model GCN2  --run_mode stage1 --split_id 0  --trial_start 0 --trial_end 0
python testing_main.py --dataset CiteSeer --model GCN2  --run_mode stage1 --split_id 0  --trial_start 0 --trial_end 0
python testing_main.py --dataset Amazon --model GCN2  --run_mode stage1 --split_id 0  --trial_start 0 --trial_end 0
python testing_main.py --dataset FacebookPagePage --normalize --model GCN2  --run_mode stage1 --split_id 0  --trial_start 0 --trial_end 0
python testing_main.py --dataset GitHub --normalize --model GCN2  --run_mode stage1 --split_id 0  --trial_start 0 --trial_end 0 

## GNN Only Look at Structure
python testing_main.py --dataset Cora --model GCN2  --run_mode stage1_structure --only_structure --structure_mode random+imp --split_id 0  --trial_start 0 --trial_end 0
python testing_main.py --dataset CiteSeer --model GCN2  --run_mode stage1_structure --only_structure --structure_mode random+imp --split_id 0  --trial_start 0 --trial_end 0
python testing_main.py --dataset Amazon --model GCN2  --run_mode stage1_structure --only_structure --structure_mode random+imp --split_id 0  --trial_start 0 --trial_end 0
python testing_main.py --dataset FacebookPagePage --normalize --model GCN2  --run_mode stage1_structure --only_structure --structure_mode random+imp --split_id 0  --trial_start 0 --trial_end 0
python testing_main.py --dataset GitHub --normalize --model GCN2  --run_mode stage1_structure --only_structure --structure_mode random+imp --split_id 0  --trial_start 0 --trial_end 0 

## MLP
python testing_main.py --dataset Cora --model MLP  --run_mode stage1_mlp --split_id 0  --trial_start 0 --trial_end 0
python testing_main.py --dataset CiteSeer --model MLP  --run_mode stage1_mlp --split_id 0  --trial_start 0 --trial_end 0
python testing_main.py --dataset Amazon --model MLP  --run_mode stage1_mlp --split_id 0  --trial_start 0 --trial_end 0
python testing_main.py --dataset FacebookPagePage --normalize --model MLP  --run_mode stage1_mlp --split_id 0  --trial_start 0 --trial_end 0
python testing_main.py --dataset GitHub --normalize --model MLP  --run_mode stage1_mlp --split_id 0  --trial_start 0 --trial_end 0



# ============================================

# + 重跑 10 次
# ============================================
# 
wait