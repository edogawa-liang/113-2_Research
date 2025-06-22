#!/bin/bash
set -e
set -x 

# 把 stage2 和 stage2_structure 從 saved_0621 直接移過來 (command 7 已跑過) 
# 但之後不小心誤刪 saved_0621.....


# ============================================
# 誤刪了: core_subgraph.... 重跑 GNNExplainer 和 Random 移除邊的實驗

# command8-1.sh


# ====================天花板 (只使用結構的GNN (移除所有特徵), MLP (Graph移除所有邊))========================
# done
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


# =================== 重新生成經過 feature to node 的解釋子圖 =========================
# 1. train base model
# ------------------------------------------------------------
## Feature to Node (含節點邊 & 特徵邊)
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_nn_fn --note feature2node_nn_fn  --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding

## Feature to Node (只有特徵邊)
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn  --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding


# 2. 生成解釋
## Feature to Node (含節點邊 & 特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
python stage2_expsubg.py --dataset GitHub --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn --feature_to_node --structure_mode random+imp 

## Feature to Node (只有特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset GitHub --normalize --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 

# ============================================
# 
wait