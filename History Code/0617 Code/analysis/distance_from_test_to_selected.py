import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from data.dataset_loader import GraphDatasetLoader

# 計算測試節點到節點的距離
# (挑選其中一個Explainer跟模型下去算, 不同解釋方法和模型選到的節點都是一樣的)
# python analysis/distance_from_test_to_selected.py --dataset Cora


class TestToNodeDistanceSaver:
    def __init__(self, base_dir, dataset, data):
        self.csv_dir = os.path.join("saved", base_dir, "GNNExplainer", dataset, "1_GCN2Classifier")
        self.record_path = os.path.join(self.csv_dir, "node_record.csv")
        self.output_path = os.path.join("saved", "analysis", "dist_from_test_to_selected", f"{dataset}.csv")
        self.data = data.clone()

        self.test_nodes = self.data.test_mask.nonzero(as_tuple=True)[0].tolist()
        self.graph = to_networkx(self.data, to_undirected=True)
        self.node_df = pd.read_csv(self.record_path)
        print(f"Read node_record from: {self.record_path}")

    def save_or_update_distances(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        strategies = [col for col in self.node_df.columns if col not in ["Node", "Explainer", "Dataset", "Model"]]
        existing = pd.DataFrame({"TestNode": self.test_nodes})
        topks = [2, 3, 5]

        for strategy in strategies:
            selected_nodes = self.node_df[self.node_df[strategy] == 1]["Node"].tolist()
            selected_set = set(selected_nodes)

            min_dists, mean_dists = [], []
            topk_dists = {k: [] for k in topks}

            for test_node in self.test_nodes:
                if test_node not in self.graph:
                    min_dists.append(np.nan)
                    mean_dists.append(np.nan)
                    for k in topks:
                        topk_dists[k].append(np.nan)
                    continue

                dist_dict = nx.single_source_shortest_path_length(self.graph, test_node)
                dists = [dist for tgt, dist in dist_dict.items() if tgt in selected_set]

                if not dists:
                    min_dists.append(np.nan)
                    mean_dists.append(np.nan)
                    for k in topks:
                        topk_dists[k].append(np.nan)
                else:
                    dists.sort()
                    min_dists.append(dists[0])
                    mean_dists.append(sum(dists) / len(dists))
                    for k in topks:
                        top_k_vals = dists[:k] if len(dists) >= k else dists
                        topk_dists[k].append(sum(top_k_vals) / len(top_k_vals))

            existing[f"{strategy}_min_dist"] = min_dists
            existing[f"{strategy}_mean_dist"] = mean_dists
            for k in topks:
                existing[f"{strategy}_top{k}_avg_dist"] = topk_dists[k]

        existing.to_csv(self.output_path, index=False)
        print(f"Saved distance file: {self.output_path}")
        return existing

    def plot_distance_summary(self, df, base_dir, dataset):
        metrics = [
            ("min_dist", "Minimum"),
            ("top2_avg_dist", "Top-2 Avg"),
            ("top3_avg_dist", "Top-3 Avg"),
            ("top5_avg_dist", "Top-5 Avg")
        ]

        strategies = sorted({
            col[: -len(metric_key) - 1]
            for col in df.columns
            for metric_key, _ in metrics
            if col.endswith(metric_key)
        })

        all_data = {label: [] for _, label in metrics}
        strategy_labels = []

        for metric_key, metric_label in metrics:
            for s in strategies:
                col = f"{s}_{metric_key}"
                if col in df.columns:
                    values = df[col].dropna().tolist()
                    all_data[metric_label].append(values)
            if not strategy_labels:
                strategy_labels = strategies

        colors = plt.cm.tab10.colors
        color_map = {s: colors[i % len(colors)] for i, s in enumerate(strategies)}

        fig, axs = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6), sharey=True)

        for ax, (metric_label, data_list) in zip(axs, all_data.items()):
            bp = ax.boxplot(data_list, patch_artist=True)
            for patch, s in zip(bp['boxes'], strategies):
                patch.set_facecolor(color_map[s])
            ax.set_title(metric_label)
            ax.set_xticks([])
            ax.yaxis.set_ticks_position('left')
            ax.set_ylabel("Distance")

        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[s]) for s in strategies]
        labels = [s.replace("_", " ").title() for s in strategies]
        fig.legend(handles, labels, loc="upper right")

        plt.suptitle("Distance from Test Nodes to Selected Nodes", fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])

        save_path = os.path.join("saved", "analysis", "dist_from_test_to_selected", f"{dataset}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved boxplot to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default= "stage2_y_edge_0.3", help="Base path to saved results")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    args = parser.parse_args()

    loader = GraphDatasetLoader(normalize=False)
    data, _, _, _, _ = loader.load_dataset(args.dataset)

    analyzer = TestToNodeDistanceSaver(
        base_dir=args.base_dir,
        dataset=args.dataset,
        data=data
    )
    df = analyzer.save_or_update_distances()
    analyzer.plot_distance_summary(df, base_dir=args.base_dir, dataset=args.dataset)
