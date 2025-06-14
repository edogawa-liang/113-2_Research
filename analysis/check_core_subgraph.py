# 檢查核心子圖的特性 (Ex: Cora CiteSeer重要的是不是都是1)

用dataset_load.py讀資料集
檢查原dataset 0 跟 1 的佔比 (所有的node_feature 即節點數量*特徵數量)

讀取
# saved/core_subgraph_mask/remove_from_GNNExplainer_edge/split_0/Amazon/{Trial}_edge_mask.npy
# saved/core_subgraph_mask/remove_from_GNNExplainer_edge/split_0/Amazon/{Trial}_feature_mask.npy

# 檢查這兩個 mask 是 1 的數量
# 檢查這個trial 應該移除的比例 (讀取 saved/remove_from_GNNExplainer_edge/split_0/result/result_0614_2303.xlsx 的fraction, fraction_feat)
# 對於nodefeature: 應該移除的數量 - mask是1的數量 就是移除是0的數量


# 傳入 remove_from_GNNExplainer_edge, dataset, Trial_start, Trial_end

匯出 原dataset 0 的數量, 原 dataset 1 的數量, 原 dataset 1 的比例, 