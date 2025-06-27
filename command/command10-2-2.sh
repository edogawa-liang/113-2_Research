#!/bin/bash
set -e
set -x 

# ==================== Stage 3 移除解釋子圖後========================

# GNNExplainer

# 3. 移除特徵 (特徵不同)
## (1) Original Graph (node mask)
### [1] random select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_difffeat --note split0_random_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_difffeat --note split0_random_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_difffeat --note split0_random_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_difffeat --note split0_random_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --run_mode stage3_random_difffeat --note split0_random_difffeat --fraction_feat 0.1 --stage1_path saved/stage1

### [2] explainer select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1

# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1

# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1

# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1

# Github 不用跑了
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1 --stage1_path saved/stage1
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_node_mask_difffeat --fraction_feat 0.1  --stage1_path saved/stage1


# ## (2) feature to Node (含節點邊 & 特徵邊)
# ### [2] explainer select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

# # python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
# # python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
# # python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn

# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
# # python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn


# ## (3) feature to Node (只有特徵邊)

# ### [2] explainer select
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn

# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn

# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn


# CFExplainer

# 3. 移除特徵 (特徵不同)
## (2) feature to Node (含節點邊 & 特徵邊)
### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn


## (3) feature to Node (只有特徵邊)
### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn
# python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_fn --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_difffeat --note split0_feat2node_difffeat_fn --fraction_feat 0.1 --feature_to_node --only_feature_node --stage1_path saved/stage1_feature2node_fn



# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose pagerank --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1 --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer  --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn
# python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --base_dir saved/stage2_feature2node_nn_fn --explainer_name GNNExplainer --node_choose all_train --run_mode stage3_GNNExplainer_difffeat --note split0_feat2node_difffeat_nn_fn --fraction_feat 0.1  --feature_to_node --stage1_path saved/stage1_feature2node_nn_fn



wait
