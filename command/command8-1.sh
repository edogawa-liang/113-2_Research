#!/bin/bash
set -e
set -x 

# ============================================

# Stage 3 移除核心子圖

# 1. 移除結構
## (1) Original Graph (edge mask)
### [1] random select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_edge --note split0_random_edge
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_edge --note split0_random_edge
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_edge --note split0_random_edge
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_edge --note split0_random_edge
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_edge --note split0_random_edge

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge
python train_remaining_main.py --dataset FacebookPagePage --normalize  --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge
python train_remaining_main.py --dataset FacebookPagePage --normalize  --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_edge --note split0_node_mask_edge

### [3] random walk select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_edge --note split0_random_walk_edge
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_edge --note split0_random_walk_edge
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_edge --note split0_random_walk_edge
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_edge --note split0_random_walk_edge
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_edge --note split0_random_walk_edge


## (2) Only Structure (只有結構下去學)
### [1] random select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_structure --note split0_random_edge_structure --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_structure --note split0_random_edge_structure --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_structure --note split0_random_edge_structure --only_structure
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_structure --note split0_random_edge_structure --only_structure
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random --fraction 0.1 --run_mode baseline_Result_structure --note split0_random_edge_structure --only_structure

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure

python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure

python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure

python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure

python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose pagerank --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2_structure --explainer_name GNNExplainer --node_choose all_train --run_mode remove_from_GNNExplainer_structure --note split0_node_mask_edge_structure --only_structure

### [3] random walk select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_structure --note split0_random_walk_edge_structure --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_structure --note split0_random_walk_edge_structure --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_structure --note split0_random_walk_edge_structure --only_structure
python train_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_structure --note split0_random_walk_edge_structure --only_structure
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type random_walk --fraction 0.1 --walk_length 10 --num_walks 3 --node_choose stratified_by_degree --run_mode remove_from_RandomWalk_structure --note split0_random_walk_edge_structure --only_structure

# ============================================

# 
wait