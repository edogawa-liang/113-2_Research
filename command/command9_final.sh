#!/bin/bash
set -e
set -x 

# ============================================

# 正在跑: command9-2, command9-7
# 還沒跑: command9-2, command9-8, command9-9

# ============================================

# 把 stage2 和 stage2_structure 從 saved_0621 直接移過來 (command 7 已跑過) 
# 但之後不小心誤刪 saved_0621.....
# 有 feature to node 記得都不要+normalize

# ============================================
# 1. 誤刪了: core_subgraph.... 重跑 GNNExplainer 和 Random 移除邊的實驗
# ------------------------------------------------------------
# command8-1.sh


# 2. Test 10 times
# ------------------------------------------------------------
## Random remove edge
python test_remaining_main.py --dataset Cora --model GCN2  --run_mode stage3_baseline_edge --split_id 0  --trial_start 0 --trial_end 1
python test_remaining_main.py --dataset CiteSeer --model GCN2  --run_mode stage3_baseline_edge --split_id 0  --trial_start 0 --trial_end 1
python test_remaining_main.py --dataset Amazon --model GCN2  --run_mode stage3_baseline_edge --split_id 0  --trial_start 0 --trial_end 1
python test_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2  --run_mode stage3_baseline_edge --split_id 0  --trial_start 0 --trial_end 1
python test_remaining_main.py --dataset GitHub --normalize --model GCN2  --run_mode stage3_baseline_edge --split_id 0  --trial_start 0 --trial_end 1    

## GNNExplainer remove edge
python test_remaining_main.py --dataset Cora --model GCN2  --run_mode stage3_GNNExplainer_edge --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset CiteSeer --model GCN2  --run_mode stage3_GNNExplainer_edge --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset Amazon --model GCN2  --run_mode stage3_GNNExplainer_edge --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2  --run_mode stage3_GNNExplainer_edge --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset GitHub --normalize --model GCN2  --run_mode stage3_GNNExplainer_edge --split_id 0  --trial_start 0 --trial_end 5    

python test_remaining_main.py --dataset Cora --model GCN2  --run_mode stage3_GNNExplainer_structure --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset CiteSeer --model GCN2  --run_mode stage3_GNNExplainer_structure --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset Amazon --model GCN2  --run_mode stage3_GNNExplainer_structure --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2  --run_mode stage3_GNNExplainer_structure --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset GitHub --normalize --model GCN2  --run_mode stage3_GNNExplainer_structure --split_id 0  --trial_start 0 --trial_end 5    

# 3. Summary Test 10 times
# ------------------------------------------------------------
## Random remove edge
python tools/result_summary.py --run_mode stage3_baseline_edge --split_id 0 --dataset Cora --trial_start 0 --trial_end 1
python tools/result_summary.py --run_mode stage3_baseline_edge --split_id 0 --dataset CiteSeer --trial_start 0 --trial_end 1
python tools/result_summary.py --run_mode stage3_baseline_edge --split_id 0 --dataset Amazon --trial_start 0 --trial_end 1
python tools/result_summary.py --run_mode stage3_baseline_edge --split_id 0 --dataset FacebookPagePage --trial_start 0 --trial_end 1
python tools/result_summary.py --run_mode stage3_baseline_edge --split_id 0 --dataset GitHub --trial_start 0 --trial_end 1  

## GNNExplainer remove edge
python tools/result_summary.py --run_mode stage3_GNNExplainer_edge --split_id 0 --dataset Cora --trial_start 0 --trial_end 5
python tools/result_summary.py --run_mode stage3_GNNExplainer_edge --split_id 0 --dataset CiteSeer --trial_start 0 --trial_end 5
python tools/result_summary.py --run_mode stage3_GNNExplainer_edge --split_id 0 --dataset Amazon --trial_start 0 --trial_end 5
python tools/result_summary.py --run_mode stage3_GNNExplainer_edge --split_id 0 --dataset FacebookPagePage --trial_start 0 --trial_end 5
python tools/result_summary.py --run_mode stage3_GNNExplainer_edge --split_id 0 --dataset GitHub --trial_start 0 --trial_end 5

python tools/result_summary.py --run_mode stage3_GNNExplainer_structure --split_id 0 --dataset Cora --trial_start 0 --trial_end 5
python tools/result_summary.py --run_mode stage3_GNNExplainer_structure --split_id 0 --dataset CiteSeer --trial_start 0 --trial_end 5
python tools/result_summary.py --run_mode stage3_GNNExplainer_structure --split_id 0 --dataset Amazon --trial_start 0 --trial_end 5
python tools/result_summary.py --run_mode stage3_GNNExplainer_structure --split_id 0 --dataset FacebookPagePage --trial_start 0 --trial_end 5
python tools/result_summary.py --run_mode stage3_GNNExplainer_structure --split_id 0 --dataset GitHub --trial_start 0 --trial_end 5 


## 


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

# 3. Summary Test 10 times
# ------------------------------------------------------------
## Original GNN
python tools/result_summary.py --run_mode stage1 --split_id 0 --dataset Cora --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1 --split_id 0 --dataset CiteSeer --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1 --split_id 0 --dataset Amazon --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1 --split_id 0 --dataset FacebookPagePage --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1 --split_id 0 --dataset GitHub --trial_start 0 --trial_end 0  

## GNN Only Look at Structure
python tools/result_summary.py --run_mode stage1_structure --split_id 0 --dataset Cora --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_structure --split_id 0 --dataset CiteSeer --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_structure --split_id 0 --dataset Amazon --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_structure --split_id 0 --dataset FacebookPagePage --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_structure --split_id 0 --dataset GitHub --trial_start 0 --trial_end 0

## MLP
python tools/result_summary.py --run_mode stage1_mlp --split_id 0 --dataset Cora --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_mlp --split_id 0 --dataset CiteSeer --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_mlp --split_id 0 --dataset Amazon --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_mlp --split_id 0 --dataset FacebookPagePage --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_mlp --split_id 0 --dataset GitHub --trial_start 0 --trial_end 0 



# =================== 重新生成經過 feature to node 的解釋子圖 =========================
# 注意不要先normalize
# 1. train base model
# ------------------------------------------------------------
## Feature to Node (含節點邊 & 特徵邊)
## commmand9-8
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_nn_fn --note feature2node_nn_fn  --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset FacebookPagePage --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset GitHub --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding

## Feature to Node (只有特徵邊)
## commmand9-8
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn  --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset FacebookPagePage --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset GitHub --model GCN2 --epochs 300 --lr 0.01 --run_mode stage1_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding


# 2. 生成解釋
## Feature to Node (含節點邊 & 特徵邊)
## commmand9-9
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
python stage2_expsubg.py --dataset FacebookPagePage --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn --feature_to_node --structure_mode random+imp 
python stage2_expsubg.py --dataset GitHub --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn --feature_to_node --structure_mode random+imp 

## Feature to Node (只有特徵邊)
## commmand9-10
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset FacebookPagePage--explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 
python stage2_expsubg.py --dataset GitHub --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node 


# ==================== CF 生成解釋 ======================== 
# 可以跑 但feature to node 學不太起來

## 原始 Graph (只有節點邊)
# command9-7
python stage2_expsubg.py --dataset Cora --explainer_type CFExplainer --run_mode stage2 --stage1_path saved/stage1
python stage2_expsubg.py --dataset CiteSeer --explainer_type CFExplainer --run_mode stage2 --stage1_path saved/stage1
python stage2_expsubg.py --dataset Amazon --explainer_type CFExplainer --run_mode stage2 --stage1_path saved/stage1
python stage2_expsubg.py --dataset FacebookPagePage  --normalize --explainer_type CFExplainer --run_mode stage2 --stage1_path saved/stage1
python stage2_expsubg.py --dataset GitHub  --normalize --explainer_type CFExplainer --run_mode stage2 --stage1_path saved/stage1

## Feature to Node (含節點邊 & 特徵邊)
# command9-11
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn 
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn
python stage2_expsubg.py --dataset FacebookPagePage --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn 
python stage2_expsubg.py --dataset GitHub --explainer_type GNNExplainer --run_mode stage2_feature2node_nn_fn --stage1_path saved/stage1_feature2node_nn_fn 

## Feature to Node (只有特徵邊)
# command9-12
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn 
python stage2_expsubg.py --dataset FacebookPagePage --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn 
python stage2_expsubg.py --dataset GitHub --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn 


# ==================== Stage 3 移除解釋子圖後========================

# GNNExplainer

# 2. 移除特徵 (特徵相同)
## (1) Original Graph (node mask)
### [1] random select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_samefeat --note split0_random_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_samefeat --note split0_random_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_samefeat --note split0_random_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_samefeat --note split0_random_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_samefeat --note split0_random_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat --stage1_path saved/stage1


## (2) feature to Node (含節點邊 & 特徵邊)

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn


## (3) feature to Node (只有特徵邊)
### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn


# 3. 移除特徵 (特徵不同)
## (1) Original Graph (node mask)
### [1] random select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_difffeat --note split0_random_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_difffeat --note split0_random_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_difffeat --note split0_random_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_difffeat --note split0_random_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_difffeat --note split0_random_difffeat --fraction_feat 0.1 --stage1_path saved/stage1

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1


## (2) feature to Node (含節點邊 & 特徵邊)
### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn


## (3) feature to Node (只有特徵邊)

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn


# 4. 移除結構+特徵 (特徵相同)
## (1) Original Graph
### [1] random select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode stage3_random_samefeat --note split0_random_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode stage3_random_samefeat --note split0_random_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode stage3_random_samefeat --note split0_random_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode stage3_random_samefeat --note split0_random_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode stage3_random_samefeat --note split0_random_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat  --stage1_path saved/stage1


## (2) feature to Node (含節點邊 & 特徵邊)
### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn



# 5. 移除結構+特徵 (特徵不同)
## (1) Original Graph
### [1] random select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode stage3_random_difffeat_edge --note split0_random_difffeat_edge --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode stage3_random_difffeat_edge --note split0_random_difffeat_edge --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode stage3_random_difffeat_edge --note split0_random_difffeat_edge --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode stage3_random_difffeat_edge --note split0_random_difffeat_edge --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode stage3_random_difffeat_edge --note split0_random_difffeat_edge --fraction_feat 0.1 --stage1_path saved/stage1

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1  --stage1_path saved/stage1
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1  --stage1_path saved/stage1

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1  --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1  --stage1_path saved/stage1

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1  --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1  --stage1_path saved/stage1

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1  --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1  --stage1_path saved/stage1

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1  --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1  --stage1_path saved/stage1


## (2) feature to Node (含節點邊 & 特徵邊)
### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn


# ==================== Stage 3 移除解釋子圖後========================

# CFExplainer

# 1. 移除結構 (節點邊)
### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize  --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.3 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.3 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.3 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize  --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.3 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.3 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1



# 2. 移除特徵 (特徵相同)
## (2) feature to Node (含節點邊 & 特徵邊)

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn


## (3) feature to Node (只有特徵邊)
### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn


# 3. 移除特徵 (特徵不同)
## (2) feature to Node (含節點邊 & 特徵邊)
### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn


## (3) feature to Node (只有特徵邊)
### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn


# 4. 移除結構+特徵 (特徵相同)
## (2) feature to Node (含節點邊 & 特徵邊)
### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn



# 5. 移除結構+特徵 (特徵不同)
## (2) feature to Node (含節點邊 & 特徵邊)
### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn


# ==================== 表格資料 ======================== 


# 
wait

