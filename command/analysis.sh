#!/bin/bash
set -e
set -x 

# 1. 測試節點到挑選節點的距離
python analysis/distance_from_test_to_selected.py --dataset Cora
python analysis/distance_from_test_to_selected.py --dataset CiteSeer
python analysis/distance_from_test_to_selected.py --dataset Amazon

# 2. 比較是否改變預測 與挑選節點距離的關聯 (都用 GNNExplainer; PageRank, stratified; Remove 20%)
# PageRank 
python analysis/dist_change_vs_unchange.py --dataset Cora  --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier --strategy top_pagerank
python analysis/dist_change_vs_unchange.py --dataset CiteSeer  --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier --strategy top_pagerank
python analysis/dist_change_vs_unchange.py --dataset Amazon  --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier --strategy top_pagerank

# stratified
python analysis/dist_change_vs_unchange.py --dataset Cora  --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier --strategy stratified_by_degree
python analysis/dist_change_vs_unchange.py --dataset CiteSeer  --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier --strategy stratified_by_degree
python analysis/dist_change_vs_unchange.py --dataset Amazon  --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier --strategy stratified_by_degree


# 3. 比較是否改變預測 測試節點在 2 hop 內的挑選節點數量 (都用 GNNExplainer; PageRank, stratified; Remove 20%)
# PageRank 
python analysis/num_test2hop_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy top_pagerank
python analysis/num_test2hop_change_vs_unchange.py --dataset CiteSeer --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy top_pagerank
python analysis/num_test2hop_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy top_pagerank

# stratified
python analysis/num_test2hop_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy stratified_by_degree
python analysis/num_test2hop_change_vs_unchange.py --dataset CiteSeer --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy stratified_by_degree
python analysis/num_test2hop_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy stratified_by_degree


# 4. 測試節點 2 hop 內的挑選節點的degree
# PageRank 
python analysis/degree_test2hop_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy top_pagerank
python analysis/degree_test2hop_change_vs_unchange.py --dataset CiteSeer --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy top_pagerank
python analysis/degree_test2hop_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy top_pagerank

# stratified
python analysis/degree_test2hop_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy stratified_by_degree
python analysis/degree_test2hop_change_vs_unchange.py --dataset CiteSeer --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy stratified_by_degree
python analysis/degree_test2hop_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --strategy stratified_by_degree


# 5. 測試節點的degree 數量
# PageRank  
python analysis/test_degree_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier
python analysis/test_degree_change_vs_unchange.py --dataset CiteSeer --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier
python analysis/test_degree_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 5_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier

# stratified
python analysis/test_degree_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier
python analysis/test_degree_change_vs_unchange.py --dataset CiteSeer --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier
python analysis/test_degree_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 17_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier


wait