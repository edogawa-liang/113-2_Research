# 檢查核心子圖的特性 (Ex: Cora CiteSeer重要的是不是都是1)

# 用dataset_load.py讀資料集
# 檢查原dataset 0 跟 1 的佔比 (所有的node_feature 即節點數量*特徵數量)

# 讀取
# # saved/core_subgraph_mask/remove_from_GNNExplainer_edge/split_0/Amazon/{Trial}_edge_mask.npy
# # saved/core_subgraph_mask/remove_from_GNNExplainer_edge/split_0/Amazon/{Trial}_feature_mask.npy

# 檢查這兩個 mask 分別是 1 的數量 = 移除1的數量
# 檢查這個trial 應該移除的比例 (讀取 saved/remove_from_GNNExplainer_edge/split_0/result/result_0614_2303.xlsx 的fraction, fraction_feat)計算數量

# 對於nodefeature: 應該移除的數量 - mask是1的數量 = 移除是0的數量


# # 傳入 remove_from_GNNExplainer_edge, split_id, dataset, Trial_start, Trial_end

# 匯出 原dataset 0 的數量, 原 dataset 1 的數量, 
# 原 dataset 1 的比例, 
# 移除1的數量, 移除0的數量, 
# 移除1的數量佔全部移除的比例, 移除1的數量佔全部1的比例, 移除0的數量佔全部0的比例

import os
import numpy as np
import pandas as pd
import argparse
from data.dataset_loader import GraphDatasetLoader

def analyze_feature_removal(args):
    run_mode = args.run_mode
    split_id = args.split_id
    dataset = args.dataset
    trial_start = args.trial_start
    trial_end = args.trial_end

    results = []

    # 讀取原始資料集的特徵矩陣 x（0/1）
    dataset_loader = GraphDatasetLoader(name=dataset)
    data = dataset_loader.load()[0]
    x = data.x.cpu().numpy()

    total_1 = int(x.sum())
    total_0 = x.size - total_1
    total_1_ratio = total_1 / x.size

    # 讀整份 result 檔案（只有一個 sheet）
    result_path = f"saved/{run_mode}/split_{split_id}/result/result_0614_2303.xlsx"
    sheet_name = f"{dataset}_remaining_graph"
    if not os.path.exists(result_path):
        print(f"[Error] result file not found: {result_path}")
        return

    df_result_all = pd.read_excel(result_path, sheet_name=sheet_name)

    for trial in range(trial_start, trial_end):
        print(f"Processing Trial {trial}...")

        base_path = f"saved/core_subgraph_mask/{run_mode}/split_{split_id}/{dataset}"
        edge_mask_path = os.path.join(base_path, f"{trial}_edge_mask.npy")
        feature_mask_path = os.path.join(base_path, f"{trial}_feature_mask.npy") 

        if not (os.path.exists(edge_mask_path) and os.path.exists(feature_mask_path)):
            print(f"  [Skip] Missing mask or original: {edge_mask_path} / {feature_mask_path}")
            continue

        feature_mask = np.load(feature_mask_path)  # shape: same as x
        edge_mask = np.load(edge_mask_path)        # 目前沒用到，可選

        total_1 = int(x.sum())
        total_0 = x.size - total_1
        total_1_ratio = total_1 / x.size

        removed_1 = int((x * feature_mask).sum())
        removed_total = int(feature_mask.sum())
        removed_0 = removed_total - removed_1

        # 找對應 trial 的 fraction_feat
        df_trial = df_result_all[df_result_all["trial"] == trial]
        if df_trial.empty:
            print(f"[Skip] Trial {trial} not found in result sheet.")
            continue
        fraction_feat = df_trial['fraction_feat'].iloc[0]
        expected_remove = int(x.size * fraction_feat)
        error = expected_remove - removed_total

        results.append({
            "Trial": trial,
            "Total 0": total_0,
            "Total 1": total_1,
            "1 ratio": round(total_1_ratio, 4),
            "Removed 1": removed_1,
            "Removed 0": removed_0,
            "Removed total": removed_total,
            "Removed 1 / removed": round(removed_1 / removed_total, 4) if removed_total > 0 else 0,
            "Removed 1 / total 1": round(removed_1 / total_1, 4) if total_1 > 0 else 0,
            "Removed 0 / total 0": round(removed_0 / total_0, 4) if total_0 > 0 else 0,
            "Expected removed": expected_remove,
            "Error": error
        })

    df = pd.DataFrame(results)
    output_path = f"{dataset}_core_subgraph_removal_summary.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved summary to {output_path}")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze core subgraph 0/1 feature removal bias")
    parser.add_argument("--run_mode", type=str, required=True, help="Run mode folder name (e.g., remove_from_GNNExplainer_edge)")
    parser.add_argument("--split_id", type=int, required=True, help="Split ID (e.g., 0)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., Amazon)")
    parser.add_argument("--trial_start", type=int, required=True, help="Start trial ID (inclusive)")
    parser.add_argument("--trial_end", type=int, required=True, help="End trial ID (exclusive)")

    args = parser.parse_args()
    analyze_feature_removal(args)
