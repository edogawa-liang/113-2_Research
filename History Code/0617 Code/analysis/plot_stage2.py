import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse

def plot_importance_from_npz(base_dir, explainer_name, dataset_name, node_choose):
    base_path = Path(base_dir) / explainer_name / dataset_name
    sub_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.endswith("GCN2Classifier")])
    node_masks = []
    edge_masks = []

    for sub_dir in sub_dirs:
        model_dir = sub_dir
        share_node_dir = model_dir / "share_nodes"
        csv_path = model_dir / "node_record.csv"

        if not csv_path.exists():
            print(f"Warning: node_record.csv not found in {model_dir}")
            continue

        df = pd.read_csv(csv_path)
        if node_choose not in df.columns:
            print(f"Warning: column '{node_choose}' not found in {csv_path}")
            continue

        selected_nodes = df[df[node_choose] == 1]["Node"].tolist()

        for node_id in selected_nodes:
            file_path = share_node_dir / f"node_{node_id}.npz"
            if not file_path.exists():
                print(f"File {file_path} does not exist. Warn!!!!.")
                continue

            data = np.load(file_path, allow_pickle=True)
            node_mask = data.get("node_mask", None)
            edge_mask = data.get("edge_mask", None)

            if node_mask is not None:
                node_masks.append(node_mask)
            if edge_mask is not None:
                edge_masks.append(edge_mask)

    if node_masks:
        # node_masks = np.stack(node_masks)  # [num_models, num_nodes, num_features]
        node_masks = np.stack(node_masks).astype(np.float64)

        feature_scores = node_masks.sum(axis=1).sum(axis=0).astype(np.float64)
        print("Feature importance scores:", feature_scores)
        print("Total feature importance sum:", np.sum(feature_scores))

        sorted_scores = np.sort(feature_scores)[::-1]
        print("Sorted feature importance scores:", sorted_scores)
        cumulative_scores = np.cumsum(sorted_scores)
        cumulative_scores /= cumulative_scores[-1]

        save_path_feat = base_path / f"node_feat_imp_dist_{node_choose}.png"
        plt.figure(figsize=(8, 6))
        x_axis_ratio = np.arange(1, len(cumulative_scores) + 1) / len(cumulative_scores)
        plt.plot(x_axis_ratio, cumulative_scores, color='teal')
        plt.xlabel("Proportion of Top Features")
        plt.ylabel("Cumulative Importance Ratio")
        plt.title("Cumulative Feature Importance Distribution")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path_feat)
        print(f"Feature importance cumulative plot saved to {save_path_feat}")
        plt.close()
    else:
        print("No node_mask found. Skipping feature importance plot.")

    if edge_masks:
        edge_masks = np.stack(edge_masks)  # [num_models, num_edges]
        edge_scores = np.sum(edge_masks, axis=0)

        nonzero_count = np.count_nonzero(edge_scores)
        print(f"Non-zero edge importance scores: {nonzero_count} / {len(edge_scores)}")

        sorted_edge_scores = np.sort(edge_scores)[::-1]
        cumulative_edge_scores = np.cumsum(sorted_edge_scores)
        cumulative_edge_scores /= cumulative_edge_scores[-1]

        save_path_edge = base_path / f"edge_imp_dist_{node_choose}.png"
        plt.figure(figsize=(8, 6))
        x_axis_ratio = np.arange(1, len(cumulative_edge_scores) + 1) / len(cumulative_edge_scores)
        plt.plot(x_axis_ratio, cumulative_edge_scores, color='royalblue')
        plt.xlabel("Proportion of Top Edges")
        plt.ylabel("Cumulative Importance Ratio")
        plt.title("Cumulative Edge Importance Distribution")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path_edge)
        print(f"Edge importance cumulative plot saved to {save_path_edge}")
        plt.close()
    else:
        print("No edge_mask found. Skipping edge importance plot.")

def parse_args():
    parser = argparse.ArgumentParser(description="Plot importance distribution from explanation outputs.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory where explanation results are stored")
    parser.add_argument("--explainer_name", type=str, required=True, help="Name of the explainer (e.g., GNNExplainer)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., Cora)")
    parser.add_argument("--node_choose", type=str, required=True, help="Node selection strategy (e.g., random, high_degree, etc.)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    plot_importance_from_npz(
        base_dir=args.base_dir,
        explainer_name=args.explainer_name,
        dataset_name=args.dataset_name,
        node_choose=args.node_choose,
    )
