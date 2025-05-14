#!/bin/bash
set -e
set -x 

# 1. train GNN only look at structure
# done
# ------------------------------------------------------------
python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure --note structure_only --use_original_label true --only_structure
python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure --note structure_only --use_original_label true --only_structure
python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_structure --note structure_only --use_original_label true --only_structure

# 2. train MLP with no feature
# done
# ------------------------------------------------------------
python training_main.py --dataset Cora --model MLP --epochs 1000 --lr 0.01 --run_mode mlp_structure --note mlp_structure_only --use_original_label true --only_structure
python training_main.py --dataset CiteSeer --model MLP --epochs 1000 --lr 0.01 --run_mode mlp_structure --note mlp_structure_only --use_original_label true --only_structure
python training_main.py --dataset Amazon --model MLP --epochs 1000 --lr 0.01 --run_mode mlp_structure --note mlp_structure_only --use_original_label true --only_structure


# - DummyExplainer
# 3. generate explanable subgraph for no feature graph, DummyExplainer, random
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type DummyExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type DummyExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset Amazon  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type DummyExplainer --stage1_path saved/original_structure --use_raw_data --only_structure

# 4. generate explanable subgraph for no feature graph, DummyExplainer, top_pagerank
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type DummyExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type DummyExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset Amazon  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type DummyExplainer --stage1_path saved/original_structure --use_raw_data --only_structure

# 5. generate explanable subgraph for no feature graph, DummyExplainer, high_betweenness
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type DummyExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type DummyExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset Amazon  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type DummyExplainer --stage1_path saved/original_structure --use_raw_data --only_structure

# 6. generate explanable subgraph for no feature graph, DummyExplainer, stratified_by_degree
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type DummyExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type DummyExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset Amazon  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type DummyExplainer --stage1_path saved/original_structure --use_raw_data --only_structure


# - GNNExplainer
# 7. generate explanable subgraph for no feature graph, GNNExplainer, random
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type GNNExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type GNNExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset Amazon  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type GNNExplainer --stage1_path saved/original_structure --use_raw_data --only_structure

# 8. generate explanable subgraph for no feature graph, GNNExplainer, top_pagerank
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type GNNExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type GNNExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset Amazon  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type GNNExplainer --stage1_path saved/original_structure --use_raw_data --only_structure

# 9. generate explanable subgraph for no feature graph, GNNExplainer, high_betweenness
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type GNNExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type GNNExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset Amazon  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type GNNExplainer --stage1_path saved/original_structure --use_raw_data --only_structure

# 10. generate explanable subgraph for no feature graph, GNNExplainer, stratified_by_degree
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type GNNExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type GNNExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset Amazon  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type GNNExplainer --stage1_path saved/original_structure --use_raw_data --only_structure

# - CFExplainer
# 7. generate explanable subgraph for no feature graph, CFExplainer, random
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type CFExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type CFExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset Amazon  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type CFExplainer --stage1_path saved/original_structure --use_raw_data --only_structure

# 8. generate explanable subgraph for no feature graph, CFExplainer, top_pagerank
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type CFExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type CFExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset Amazon  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type CFExplainer --stage1_path saved/original_structure --use_raw_data --only_structure

# 9. generate explanable subgraph for no feature graph, CFExplainer, high_betweenness
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type CFExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type CFExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset Amazon  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type CFExplainer --stage1_path saved/original_structure --use_raw_data --only_structure

# 10. generate explanable subgraph for no feature graph, CFExplainer, stratified_by_degree
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type CFExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type CFExplainer --stage1_path saved/original_structure --use_raw_data --only_structure
python stage2_expsubg.py --dataset Amazon  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_structure --explainer_type CFExplainer --stage1_path saved/original_structure --use_raw_data --only_structure


# 11. node share for all explainers
# ------------------------------------------------------------
python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_structure --explainer DummyExplainer --dataset Cora
python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_structure --explainer DummyExplainer --dataset CiteSeer
python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_structure --explainer DummyExplainer --dataset Amazon

python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_structure --explainer GNNExplainer --dataset Cora
python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_structure --explainer GNNExplainer --dataset CiteSeer
python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_structure --explainer GNNExplainer --dataset Amazon

python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_structure --explainer CFExplainer --dataset Cora
python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_structure --explainer CFExplainer --dataset CiteSeer
python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_structure --explainer CFExplainer --dataset Amazon


# - train GNN 
# 12. Train GNN model from remove random edge (only structure)
# ------------------------------------------------------------
# - fraction = 0.1
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.1 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.1 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.1 --only_structure

# - fraction = 0.2
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.2 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.2 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.2 --only_structure

# - fraction = 0.3
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.3 --only_structure

# - fraction = 0.4
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.4 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.4 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.4 --only_structure

# - fraction = 0.5
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.5 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.5 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_structure --note random_structure --selector_type random --fraction 0.5 --only_structure



# 13. Train GNN model from remove random walk (only structure)
# ------------------------------------------------------------
# - fraction = 0.1
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.1  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.1  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.1  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.1  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.1  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.1  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.1  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.1  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.1  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.1  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.1  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.1  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure

# - fraction = 0.2
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure

# - fraction = 0.3
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.3  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.3  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.3  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.3  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.3  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.3  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.3  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.3  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.3  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.3  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.3  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.3  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure

# - fraction = 0.4
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.4  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.4  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.4  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.4  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.4  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.4  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.4  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.4  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.4  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.4  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.4  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.4  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure

# - fraction = 0.5
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.5  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.5  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note random_RandomWalk_structure --selector_type random_walk --fraction 0.5  --node_choose random --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.5  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.5  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note top_pagerank_RandomWalk_structure --selector_type random_walk --fraction 0.5  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.5  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.5  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note high_betweenness_RandomWalk_structure --selector_type random_walk --fraction 0.5  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.5  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.5  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_structure --note stratified_by_degree_RandomWalk_structure --selector_type random_walk --fraction 0.5  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --only_structure


# 14. Train GNN model from remove DummyExplainer edge (only structure)
# ------------------------------------------------------------
# - fraction 0.1
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure

# - fraction 0.2
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure

# - fraction 0.3
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure

# - fraction 0.4
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure

# - fraction 0.5
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_random_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose random --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_top_pagerank_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose top_pagerank --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_high_betweenness_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose high_betweenness --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_structure --note true_y_DummyExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name DummyExplainer --node_choose stratified_by_degree --only_structure


# 15. Train GNN model from remove GNNExplainer edge (only structure)
# ------------------------------------------------------------
# - fraction 0.1
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.1 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure

# - fraction 0.2
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure

# - fraction 0.3
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.3 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure

# - fraction 0.4
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.4 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure

# - fraction 0.5
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_random_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose random --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_top_pagerank_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose top_pagerank --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_high_betweenness_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose high_betweenness --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_structure --note true_y_GNNExplainer_stratified_by_degree_structure --selector_type explainer --fraction 0.5 --base_dir saved/stage2_y_edge_0.3_structure --explainer_name GNNExplainer --node_choose stratified_by_degree --only_structure


# 16. Train GNN model from remove CFExplainer edge (only structure)
# ------------------------------------------------------------
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_CFExplainer_structure --note true_y_CFExplainer_random_structure --selector_type explainer --base_dir saved/stage2_y_edge_0.3_structure --explainer_name CFExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_CFExplainer_structure --note true_y_CFExplainer_random_structure --selector_type explainer --base_dir saved/stage2_y_edge_0.3_structure --explainer_name CFExplainer --node_choose random --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_CFExplainer_structure --note true_y_CFExplainer_random_structure --selector_type explainer --base_dir saved/stage2_y_edge_0.3_structure --explainer_name CFExplainer --node_choose random --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_CFExplainer_structure --note true_y_CFExplainer_top_pagerank_structure --selector_type explainer --base_dir saved/stage2_y_edge_0.3_structure --explainer_name CFExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_CFExplainer_structure --note true_y_CFExplainer_top_pagerank_structure --selector_type explainer --base_dir saved/stage2_y_edge_0.3_structure --explainer_name CFExplainer --node_choose top_pagerank --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_CFExplainer_structure --note true_y_CFExplainer_top_pagerank_structure --selector_type explainer --base_dir saved/stage2_y_edge_0.3_structure --explainer_name CFExplainer --node_choose top_pagerank --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_CFExplainer_structure --note true_y_CFExplainer_high_betweenness_structure --selector_type explainer --base_dir saved/stage2_y_edge_0.3_structure --explainer_name CFExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_CFExplainer_structure --note true_y_CFExplainer_high_betweenness_structure --selector_type explainer --base_dir saved/stage2_y_edge_0.3_structure --explainer_name CFExplainer --node_choose high_betweenness --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_CFExplainer_structure --note true_y_CFExplainer_high_betweenness_structure --selector_type explainer --base_dir saved/stage2_y_edge_0.3_structure --explainer_name CFExplainer --node_choose high_betweenness --only_structure

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_CFExplainer_structure --note true_y_CFExplainer_stratified_by_degree_structure --selector_type explainer --base_dir saved/stage2_y_edge_0.3_structure --explainer_name CFExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_CFExplainer_structure --note true_y_CFExplainer_stratified_by_degree_structure --selector_type explainer --base_dir saved/stage2_y_edge_0.3_structure --explainer_name CFExplainer --node_choose stratified_by_degree --only_structure
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_CFExplainer_structure --note true_y_CFExplainer_stratified_by_degree_structure --selector_type explainer --base_dir saved/stage2_y_edge_0.3_structure --explainer_name CFExplainer --node_choose stratified_by_degree --only_structure
