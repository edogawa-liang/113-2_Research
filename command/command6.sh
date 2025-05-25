#!/bin/bash
set -e
set -x 

# 核心子圖包含特徵
# feature2node
# 將特徵變成節點，一起訓練 GNN 模型並生成解釋子圖

# # 1. 將特徵變成節點，訓練 Stage1 的 GNN 模型
# # done
# # ------------------------------------------------------------
# python training_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node --note feature2node --use_original_label true --feature_to_node
# python training_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node --note feature2node --use_original_label true --feature_to_node
# python training_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode original_feature2node --note feature2node --use_original_label true --feature_to_node


# # Stage2 生成解釋子圖

# # - DummyExplainer
# # 3. generate explanable subgraph for feature to node graph, DummyExplainer, random
# # done
# # ------------------------------------------------------------
# python stage2_expsubg.py --dataset Cora  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type DummyExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset CiteSeer  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type DummyExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset Amazon  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type DummyExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node

# # 4. generate explanable subgraph for feature to node graph, DummyExplainer, top_pagerank
# # done
# # ------------------------------------------------------------
# python stage2_expsubg.py --dataset Cora  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type DummyExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset CiteSeer  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type DummyExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset Amazon  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type DummyExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node

# # 5. generate explanable subgraph for feature to node graph, DummyExplainer, high_betweenness
# # done
# # ------------------------------------------------------------
# python stage2_expsubg.py --dataset Cora  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type DummyExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset CiteSeer  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type DummyExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset Amazon  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type DummyExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node

# # 6. generate explanable subgraph for feature to node graph, DummyExplainer, stratified_by_degree
# # done
# # ------------------------------------------------------------
# python stage2_expsubg.py --dataset Cora  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type DummyExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset CiteSeer  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type DummyExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset Amazon  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type DummyExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node


# # - GNNExplainer
# # 7. generate explanable subgraph for feature to node graph, GNNExplainer, random
# # done
# # ------------------------------------------------------------
# python stage2_expsubg.py --dataset Cora  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type GNNExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset CiteSeer  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type GNNExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset Amazon  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type GNNExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node

# # 8. generate explanable subgraph for feature to node graph, GNNExplainer, top_pagerank
# # done
# # ------------------------------------------------------------
# python stage2_expsubg.py --dataset Cora  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type GNNExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset CiteSeer  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type GNNExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset Amazon  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type GNNExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node

# # 9. generate explanable subgraph for feature to node graph, GNNExplainer, high_betweenness
# # done
# # ------------------------------------------------------------
# python stage2_expsubg.py --dataset Cora  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type GNNExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset CiteSeer  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type GNNExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset Amazon  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type GNNExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node

# # 10. generate explanable subgraph for feature to node graph, GNNExplainer, stratified_by_degree
# # done
# # ------------------------------------------------------------
# python stage2_expsubg.py --dataset Cora  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type GNNExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset CiteSeer  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type GNNExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
# python stage2_expsubg.py --dataset Amazon  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type GNNExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node



# # 11. node share for all explainers
# # ------------------------------------------------------------
# python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_feature2node --explainer DummyExplainer --dataset Cora
# python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_feature2node --explainer DummyExplainer --dataset CiteSeer
# python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_feature2node --explainer DummyExplainer --dataset Amazon

# python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_feature2node --explainer GNNExplainer --dataset Cora
# python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_feature2node --explainer GNNExplainer --dataset CiteSeer
# python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_feature2node --explainer GNNExplainer --dataset Amazon


# Start Stage 3 (先固定移除20%邊)
# fraction_feat=0.1

# train remove random edge and random feature
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.1 --same_feat True
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.1 --same_feat True
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.1 --same_feat True


# train remove randomwalk
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --feature_to_node 
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --feature_to_node  &

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --feature_to_node &


# train remove DummyExplainer (不管有沒有 feature2node，移除的 feature 都是 random)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.1

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.1

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.1

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.1



# train remove GNNExplainer (node_imp)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.1

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.1

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.1

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.1


# train remove GNNExplainer (feat2node)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.1 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.1 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.1 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.1 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.1 --feature_to_node


---
# Start Stage 3 (先固定移除20%邊)
# fraction_feat=0.2

# train remove random edge and random feature
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.2  --same_feat True
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.2  --same_feat True
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.2  --same_feat True


# train remove randomwalk
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2 --feature_to_node &

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2 --feature_to_node &


# train remove DummyExplainer (不管有沒有 feature2node，移除的 feature 都是 random)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.2
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.2
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.2

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.2
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.2
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.2

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.2
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.2
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.2

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.2
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.2
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.2



# train remove GNNExplainer (node_imp)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.2
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.2
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.2

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.2
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.2
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.2

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.2
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.2
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.2

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.2
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.2
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.2


# train remove GNNExplainer (feat2node)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.2 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.2 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.2 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.2 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.2 --feature_to_node


---
# Start Stage 3 (先固定移除20%邊)
# fraction_feat=0.3

# train remove random edge and random feature
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.3  --same_feat True
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.3  --same_feat True
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.3  --same_feat True


# train remove randomwalk
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3 --feature_to_node  &

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3 --feature_to_node  &


# train remove DummyExplainer (不管有沒有 feature2node，移除的 feature 都是 random)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.3
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.3
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.3

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.3
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.3
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.3

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.3
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.3
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.3

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.3
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.3
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.3



# train remove GNNExplainer (node_imp)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.3
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.3
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.3

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.3
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.3
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.3

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.3
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.3
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.3

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.3
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.3
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.3


# train remove GNNExplainer (feat2node)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.3 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.3 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.3 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.3 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.3 --feature_to_node


---
# # Start Stage 3 (先固定移除20%邊)
# # fraction_feat=0.4

# # train remove random edge and random feature
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.4 --same_feat True
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.4  --same_feat True
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.4  --same_feat True


# # train remove randomwalk
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.4 --feature_to_node

# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.4 --feature_to_node

# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.4 --feature_to_node  &

# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.4 --feature_to_node  &


# # train remove DummyExplainer (不管有沒有 feature2node，移除的 feature 都是 random)
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.4
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.4
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.4

# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.4
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.4
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.4

# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.4
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.4
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.4

# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.4
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.4
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.4



# # train remove GNNExplainer (node_imp)
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.4
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.4
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.4

# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.4
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.4
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.4

# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.4
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.4
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.4

# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.4
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.4
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.4


# # train remove GNNExplainer (feat2node)
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.4 --feature_to_node

# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.4 --feature_to_node

# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.4 --feature_to_node

# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.4 --feature_to_node
# python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.4 --feature_to_node


---
# Start Stage 3 (先固定移除20%邊)
# fraction_feat=0.5

# train remove random edge and random feature
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.5  --same_feat True
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.5  --same_feat True
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult_randfeat --note randfeat --selector_type random --fraction 0.2 --fraction_feat 0.5 --same_feat True


# train remove randomwalk
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note random_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.5 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note top_pagerank_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.5 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note high_betweenness_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.5 --feature_to_node  &

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_randfeat --note stratified_by_degree_RandomWalk_randfeat --selector_type random_walk --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.5 --feature_to_node  &


# train remove DummyExplainer (不管有沒有 feature2node，移除的 feature 都是 random)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.5
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.5
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose random --fraction_feat 0.5

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.5
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.5
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose top_pagerank --fraction_feat 0.5

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.5
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.5
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose high_betweenness --fraction_feat 0.5

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.5
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.5
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_DummyExplainer_randfeat --note true_y_DummyExplainer_randfeat --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name DummyExplainer --node_choose stratified_by_degree --fraction_feat 0.5



# train remove GNNExplainer (node_imp)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.5
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.5
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction_feat 0.5

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.5
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.5
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.5

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.5
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.5
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.5

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.5
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.5
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_node_mask --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.5


# train remove GNNExplainer (feat2node)
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose random --fraction_feat 0.5 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose top_pagerank --fraction_feat 0.5 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose high_betweenness --fraction_feat 0.5 --feature_to_node

python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.5 --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_GNNExplainer_randfeat --note true_y_GNNExplainer_feat2node --selector_type explainer --fraction 0.2 --base_dir saved/stage2_y_edge_0.3_feature2node --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction_feat 0.5 --feature_to_node


wait