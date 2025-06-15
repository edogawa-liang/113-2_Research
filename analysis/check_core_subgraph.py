# 檢查核心子圖的特性 (Ex: Cora CiteSeer重要的是不是都是1)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import numpy as np
import pandas as pd
import argparse
from data.dataset_loader import GraphDatasetLoader
import glob

def analyze_feature_removal(args):
    run_mode = args.run_mode
    split_id = args.split_id
    dataset = args.dataset
    trial_start = args.trial_start
    trial_end = args.trial_end

    results = []

    # 讀取原始資料集的特徵矩陣 x（0/1）
    loader = GraphDatasetLoader()
    data, _, _, _, _ = loader.load_dataset(args.dataset)
    x = data.x.cpu().numpy()

    total_1 = int(x.sum()) # 原始資料中是 1 的數量
    total_0 = x.size - total_1 # 原始資料中是 0 的數量
    total_1_ratio = total_1 / x.size

    # 讀整份 result 檔案（只有一個 sheet）
    result_path = glob.glob(f"saved/{run_mode}/split_{split_id}/result/*.xlsx")[0]

    sheet_name = f"{dataset}_remaining_graph"
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
        edge_mask = np.load(edge_mask_path)        # 目前沒用到

        # 移除的1的數量
        removed_1 = int(feature_mask.sum()) # 核心子圖只會存下移除1的部分

        # 找對應 trial 的 fraction_feat
        df_trial = df_result_all[df_result_all["Trial"] == trial]
        if df_trial.empty:
            print(f"[Skip] Trial {trial} not found in result sheet.")
            continue
        fraction_feat = df_trial['fraction_feat'].iloc[0]
        feature_to_node = str(df_trial['feature_to_node'].iloc[0]).lower() == 'true' if 'feature_to_node' in df_trial.columns else False
        
        same_feat = "same_feat" in run_mode


        # feature to node
        if feature_to_node:
            if same_feat:
                expected_remove = int(x.size * fraction_feat)
            else: # 核心子圖只會選到1
                expected_remove = int(total_1 * fraction_feat) # removed_0 理論上他要是0
        
        else: # node_mask
            expected_remove = int(x.size * fraction_feat)

        removed_0 = expected_remove - removed_1

        results.append({
            "Trial": trial,
            "Total 0": total_0,
            "Total 1": total_1,
            "1 ratio": round(total_1_ratio, 4),
            "Removed 1": removed_1,
            "Removed 0": removed_0,
            "Removed total": expected_remove,
            "Fraction feat": fraction_feat,
            "Removed 1 / removed": round(removed_1 / expected_remove, 4) if expected_remove > 0 else 0,
            "Removed 1 / total 1": round(removed_1 / total_1, 4) if total_1 > 0 else 0,
            "Removed 0 / total 0": round(removed_0 / total_0, 4) if total_0 > 0 else 0,
            "Expected removed": expected_remove,
        })

    df = pd.DataFrame(results)

    output_dir = f"saved/core_subgraph_summary/{run_mode}/split_{split_id}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset}_core_subgraph_summary.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved summary to {output_path}")
    print(df)

# Usage: python analysis/check_core_subgraph.py --run_mode remove_from_GNNExplainer_samefeat --split_id 0 --dataset Cora --trial_start 0 --trial_end 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze core subgraph 0/1 feature removal bias")
    parser.add_argument("--run_mode", type=str, required=True, help="Run mode folder name (e.g., remove_from_GNNExplainer_edge)")
    parser.add_argument("--split_id", type=int, default=0, help="Split ID (e.g., 0)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., Amazon)")
    parser.add_argument("--trial_start", type=int, required=True, help="Start trial ID (inclusive)")
    parser.add_argument("--trial_end", type=int, required=True, help="End trial ID (exclusive)")

    args = parser.parse_args()
    analyze_feature_removal(args)
