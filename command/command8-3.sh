#!/bin/bash
set -e
set -x 

# ============================================

# Stage 3 移除核心子圖


# # 3. 移除特徵 (特徵不同)(10%特徵)
# ## (1) Original Graph (node mask)
# ### [1] random select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.1
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.1
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.1
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.1
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.1

# ### [2] explainer select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 

# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 

# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 

# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 

# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 

### [3] random walk select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat --fraction_feat 0.1 


## (2) feature to Node (含節點邊 & 特徵邊)
### [1] random select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.1 

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node

# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node

### [3] random walk select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node


## (3) feature to Node (只有特徵邊)
### [1] random select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.1 --same_feat --only_feature_node

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node

# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node

### [3] random walk select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node


# ============================================

# # 3. 移除特徵 (特徵不同)(30%特徵)
# ## (1) Original Graph (node mask)
# ### [1] random select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.3
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.3
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.3
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.3
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.3

# ### [2] explainer select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3 
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3 

# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3 
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3 

# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3 
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3 

# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3 
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3 

# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3 
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.3 

# ### [3] random walk select
# # python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat --fraction_feat 0.3 


# ## (2) feature to Node (含節點邊 & 特徵邊)
# ### [1] random select
# # python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.3 

# ### [2] explainer select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3 --feature_to_node
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3  --feature_to_node
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3  --feature_to_node

# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3 --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3  --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3  --feature_to_node

# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3 --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3  --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3  --feature_to_node

# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3 --feature_to_node
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3  --feature_to_node
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3  --feature_to_node

# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3 --feature_to_node
# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3  --feature_to_node
# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.3  --feature_to_node

# ### [3] random walk select
# # python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_nn_fn --fraction_feat 0.3 --feature_to_node
# # python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_nn_fn --fraction_feat 0.3 --feature_to_node
# # python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_nn_fn --fraction_feat 0.3 --feature_to_node
# # python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_nn_fn --fraction_feat 0.3 --feature_to_node
# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_nn_fn --fraction_feat 0.3 --feature_to_node


# ## (3) feature to Node (只有特徵邊)
# ### [1] random select
# # python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.3 --same_feat --only_feature_node

# ### [2] explainer select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node

# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node

# # python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node

# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node

# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node

# ### [3] random walk select
# # python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat_fn --fraction_feat 0.3 --feature_to_node --only_feature_node


# ============================================

# Stage 3 移除核心子圖

# 5. 移除結構+特徵 (特徵不同)(10%邊, 10%特徵)
## (1) Original Graph
### [1] random select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_difffeat_edge --note split0_random_difffeat_edge --fraction_feat 0.1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_difffeat_edge --note split0_random_difffeat_edge --fraction_feat 0.1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_difffeat_edge --note split0_random_difffeat_edge --fraction_feat 0.1
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_difffeat_edge --note split0_random_difffeat_edge --fraction_feat 0.1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_difffeat_edge --note split0_random_difffeat_edge --fraction_feat 0.1

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_node_mask_difffeat_edge --fraction_feat 0.1 

### [3] random walk select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat --note split0_random_walk_difffeat --fraction_feat 0.1 



## (2) feature to Node (含節點邊 & 特徵邊)
### [1] random select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_difffeat --note split0_random_difffeat --fraction_feat 0.1 

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node

# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node

### [3] random walk select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat_edge --note split0_random_walk_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat_edge --note split0_random_walk_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat_edge --note split0_random_walk_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat_edge --note split0_random_walk_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_difffeat_edge --note split0_random_walk_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node



## (3) Only Structure + Feature to Node (只有特徵邊) 還沒做
### [1] random select
### [2] explainer select
### [3] random walk select
#


python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node


python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_difffeat_edge --note split0_feat2node_difffeat_edge_nn_fn --fraction_feat 0.1  --feature_to_node


# 
wait