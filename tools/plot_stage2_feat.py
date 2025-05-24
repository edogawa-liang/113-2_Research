import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_feature_importance_from_npz(base_dir, explainer_name, dataset_name, node_choose, top_k_percent_feat=0, use_feature_to_node=False, save_path="feature_importance_distribution.png"):
    base_path = Path(base_dir) / explainer_name / dataset_name
    sub_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.endswith("GCN2Classifier")])
    node_masks = []

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
            if not use_feature_to_node and top_k_percent_feat != 0:
                node_mask = data.get("node_mask", None)
                if node_mask is not None:
                    node_masks.append(node_mask)

    if not node_masks:
        print("No node_mask found. Nothing to plot.")
        return

    node_masks = np.stack(node_masks)  # [num_models, num_nodes, num_features]
    feature_scores = node_masks.sum(axis=1).sum(axis=0)

    plt.figure(figsize=(8, 6))
    plt.hist(feature_scores, bins=50, color='teal', alpha=0.7, edgecolor='black')
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Frequency")
    plt.title("Feature Importance Score Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Feature importance histogram saved to {save_path}")
    plt.show()

# Example usage:
# plot_feature_importance_from_npz(
#     base_dir="saved/stage2_node_ratio_0.01",
#     explainer_name="GNNExplainer",
#     dataset_name="Cora",
#     node_choose="random",
#     top_k_percent_feat=0.3,
#     use_feature_to_node=False
# )