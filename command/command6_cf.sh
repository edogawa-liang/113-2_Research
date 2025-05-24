#!/bin/bash
set -e
set -x 

# feature2node
# 將特徵變成節點，一起訓練 GNN 模型並生成解釋子圖
# CF 的 stage2 和 train_remain 都還沒修改


# - CFExplainer
# 7. generate explanable subgraph for feature to node graph, CFExplainer, random
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type CFExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type CFExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
python stage2_expsubg.py --dataset Amazon  --choose_nodes random --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type CFExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node

# 8. generate explanable subgraph for feature to node graph, CFExplainer, top_pagerank
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type CFExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type CFExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
python stage2_expsubg.py --dataset Amazon  --choose_nodes top_pagerank --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type CFExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node

# 9. generate explanable subgraph for feature to node graph, CFExplainer, high_betweenness
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type CFExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type CFExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
python stage2_expsubg.py --dataset Amazon  --choose_nodes high_betweenness --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type CFExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node

# 10. generate explanable subgraph for feature to node graph, CFExplainer, stratified_by_degree
# done
# ------------------------------------------------------------
python stage2_expsubg.py --dataset Cora  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type CFExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
python stage2_expsubg.py --dataset CiteSeer  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type CFExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node
python stage2_expsubg.py --dataset Amazon  --choose_nodes stratified_by_degree --node_ratio auto --edge_ratio 0.3 --run_mode stage2_y_edge_0.3_feature2node --explainer_type CFExplainer --stage1_path saved/original_feature2node --use_raw_data --feature_to_node


# 11. node share for all explainers
# ------------------------------------------------------------
python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_feature2node --explainer CFExplainer --dataset Cora
python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_feature2node --explainer CFExplainer --dataset CiteSeer
python tools/stage2_node_share.py --stage_dir stage2_y_edge_0.3_feature2node --explainer CFExplainer --dataset Amazon



wait