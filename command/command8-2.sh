#!/bin/bash
set -e
set -x 


# Stage 3 移除核心子圖

# # 2. 移除特徵 (特徵相同)(10%特徵)
# ## (1) Original Graph (node mask)
# ### [1] random select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.1 --same_feat
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.1 --same_feat
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.1 --same_feat
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.1 --same_feat
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.1 --same_feat

# ### [2] explainer select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat

# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat

# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat

# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat

# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.1 --same_feat

### [3] random walk select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat --fraction_feat 0.1 --same_feat


## (2) feature to Node (含節點邊 & 特徵邊)
### [1] random select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.1 --same_feat

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node

### [3] random walk select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node


## (3) feature to Node (只有特徵邊)
### [1] random select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.1 --same_feat --only_feature_node

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node

### [3] random walk select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_fn --fraction_feat 0.1 --same_feat --feature_to_node --only_feature_node

# ============================================

# # 2. 移除特徵 (特徵相同)(30%特徵)
# ## (1) Original Graph (node mask)
# ### [1] random select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.3 --same_feat
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.3 --same_feat
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.3 --same_feat
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.3 --same_feat
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.3 --same_feat

# ### [2] explainer select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat

# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat

# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat

# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat

# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_node_mask_samefeat --fraction_feat 0.3 --same_feat

# ### [3] random walk select
# # python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat --fraction_feat 0.3 --same_feat


# ## (2) feature to Node (含節點邊 & 特徵邊)
# ### [1] random select
# # python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.3 --same_feat

# ### [2] explainer select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node

# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node

# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node

# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node

# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node
# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node
# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node

# ### [3] random walk select
# # python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node
# # python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node
# # python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node
# # python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node
# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_nn_fn --fraction_feat 0.3 --same_feat --feature_to_node


# ## (3) feature to Node (只有特徵邊)
# ### [1] random select
# # python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.3 --same_feat --only_feature_node

# ### [2] explainer select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node

# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node

# # python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node

# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node

# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat --note split0_feat2node_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node

# ### [3] random walk select
# # python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node
# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat_fn --fraction_feat 0.3 --same_feat --feature_to_node --only_feature_node

# # ============================================



# command 8-4

# Stage 3 移除核心子圖

# 4. 移除結構+特徵 (特徵相同)(10%邊, 10%特徵)
## (1) Original Graph
### [1] random select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_samefeat --note split0_random_samefeat_edge --fraction_feat 0.1 --same_feat
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_samefeat --note split0_random_samefeat_edge --fraction_feat 0.1 --same_feat
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_samefeat --note split0_random_samefeat_edge --fraction_feat 0.1 --same_feat
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_samefeat --note split0_random_samefeat_edge --fraction_feat 0.1 --same_feat
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_samefeat --note split0_random_samefeat_edge --fraction_feat 0.1 --same_feat

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_node_mask_samefeat_edge --fraction_feat 0.1 --same_feat

### [3] random walk select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat --note split0_random_walk_samefeat --fraction_feat 0.1 --same_feat


## (2) feature to Node (含節點邊 & 特徵邊)
### [1] random select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_samefeat --note split0_random_samefeat --fraction_feat 0.1 --same_feat

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_samefeat_edge --note split0_feat2node_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node

### [3] random walk select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat_edge --note split0_random_walk_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 30 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat_edge --note split0_random_walk_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat_edge --note split0_random_walk_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat_edge --note split0_random_walk_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 300 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_samefeat_edge --note split0_random_walk_samefeat_edge_nn_fn --fraction_feat 0.1 --same_feat --feature_to_node


## (3) Only Structure + Feature to Node (只有特徵邊) 還沒做
### [1] random select
### [2] explainer select
### [3] random walk select

# ============================================

# 
wait