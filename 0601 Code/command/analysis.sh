#!/bin/bash
set -e
set -x 

# 1. 測試節點到挑選節點的距離
python analysis/distance_from_test_to_selected.py --dataset Cora
python analysis/distance_from_test_to_selected.py --dataset CiteSeer
python analysis/distance_from_test_to_selected.py --dataset Amazon

# 2. 比較是否改變預測 與挑選節點距離的關聯 (都用 GNNExplainer; PageRank, stratified; Remove 20%) 
# PageRank 
python analysis/dist_change_vs_unchange.py --dataset Cora  --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier --strategy top_pagerank  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0
python analysis/dist_change_vs_unchange.py --dataset CiteSeer  --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier --strategy top_pagerank  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0
python analysis/dist_change_vs_unchange.py --dataset Amazon  --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier --strategy top_pagerank  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0

# stratified
python analysis/dist_change_vs_unchange.py --dataset Cora  --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier --strategy stratified_by_degree  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0
python analysis/dist_change_vs_unchange.py --dataset CiteSeer  --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier --strategy stratified_by_degree  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0
python analysis/dist_change_vs_unchange.py --dataset Amazon  --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier --strategy stratified_by_degree  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0


# 3. 比較是否改變預測 測試節點在 2 hop 內的挑選節點數量 (都用 GNNExplainer; PageRank, stratified; Remove 20%)
# PageRank 
python analysis/num_test2hop_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy top_pagerank  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0
python analysis/num_test2hop_change_vs_unchange.py --dataset CiteSeer --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy top_pagerank  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0
python analysis/num_test2hop_change_vs_unchange.py --dataset Amazon --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy top_pagerank  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0

# stratified
python analysis/num_test2hop_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy stratified_by_degree  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0
python analysis/num_test2hop_change_vs_unchange.py --dataset CiteSeer --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy stratified_by_degree  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0
python analysis/num_test2hop_change_vs_unchange.py --dataset Amazon --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy stratified_by_degree  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0


# 4. 測試節點 2 hop 內的挑選節點的degree 
# PageRank 
python analysis/degree_test2hop_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy top_pagerank --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0
python analysis/degree_test2hop_change_vs_unchange.py --dataset CiteSeer --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy top_pagerank --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0
python analysis/degree_test2hop_change_vs_unchange.py --dataset Amazon --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy top_pagerank --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0

# stratified
python analysis/degree_test2hop_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy stratified_by_degree --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0
python analysis/degree_test2hop_change_vs_unchange.py --dataset CiteSeer --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy stratified_by_degree --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0
python analysis/degree_test2hop_change_vs_unchange.py --dataset Amazon --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy stratified_by_degree --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0


# 5. 測試節點的degree 數量
# PageRank  
python analysis/test_degree_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0 --strategy top_pagerank
python analysis/test_degree_change_vs_unchange.py --dataset CiteSeer --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0 --strategy top_pagerank
python analysis/test_degree_change_vs_unchange.py --dataset Amazon --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0 --strategy top_pagerank

# stratified
python analysis/test_degree_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0 --strategy stratified_by_degree
python analysis/test_degree_change_vs_unchange.py --dataset CiteSeer --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0 --strategy stratified_by_degree
python analysis/test_degree_change_vs_unchange.py --dataset Amazon --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0 --strategy stratified_by_degree


# 6. 測試模型預測結果的改變 
# PageRank  
python analysis/check_pred_result.py --dataset Cora --original_run_mode original --original_model 1_GCN2Classifier --new_run_mode remove_from_GNNExplainer --new_model 5_GCN2Classifier --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction 0.2 --fraction_feat 0.0 
python analysis/check_pred_result.py --dataset CiteSeer --original_run_mode original --original_model 1_GCN2Classifier --new_run_mode remove_from_GNNExplainer --new_model 5_GCN2Classifier --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction 0.2 --fraction_feat 0.0 
python analysis/check_pred_result.py --dataset Amazon --original_run_mode original --original_model 1_GCN2Classifier --new_run_mode remove_from_GNNExplainer --new_model 5_GCN2Classifier --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose top_pagerank --fraction 0.2 --fraction_feat 0.0 

# stratified
python analysis/check_pred_result.py --dataset Cora --original_run_mode original --original_model 1_GCN2Classifier --new_run_mode remove_from_GNNExplainer --new_model 17_GCN2Classifier --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction 0.2 --fraction_feat 0.0 
python analysis/check_pred_result.py --dataset CiteSeer --original_run_mode original --original_model 1_GCN2Classifier --new_run_mode remove_from_GNNExplainer --new_model 17_GCN2Classifier --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction 0.2 --fraction_feat 0.0 
python analysis/check_pred_result.py --dataset Amazon --original_run_mode original --original_model 1_GCN2Classifier --new_run_mode remove_from_GNNExplainer --new_model 17_GCN2Classifier --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose stratified_by_degree --fraction 0.2 --fraction_feat 0.0 

wait