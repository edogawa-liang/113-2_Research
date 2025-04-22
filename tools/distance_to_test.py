# calculate the distance between choose node and test node
# 1_GCN2Classifier, 2_GCN2Classifier, 3_GCN2Classifier.....只要使用同一種節點選取方式，選出來的節點都會一樣，因此只要算 1_GCN2Classifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from torch_geometric.utils import to_networkx
import networkx as nx
from data.dataset_loader import GraphDatasetLoader
import argparse
import matplotlib.pyplot as plt
import numpy as np

class NodeDistanceToTestSaver:
    def __init__(self, csv_dir, data, k=3):
        self.csv_dir = csv_dir
        self.record_path = os.path.join(csv_dir, "node_record.csv")
        self.output_path = os.path.join(csv_dir, "distance_to_test.csv")
        self.data = data.clone()
        self.k = k

        self.test_nodes = self.data.test_mask.nonzero(as_tuple=True)[0].tolist()
        self.graph = to_networkx(self.data, to_undirected=True)
        self.node_df = pd.read_csv(self.record_path)
        print(f"Read the nodes information from: {self.record_path}")


    def _compute_distances_for_nodes(self, nodes):
        test_nodes_set = set(self.test_nodes)
        results = []
        for node in nodes:
            if node not in self.graph:
                results.append((np.nan, np.nan, np.nan))
                continue
            dist_dict = nx.single_source_shortest_path_length(self.graph, source=node)
            dists = [dist for tgt, dist in dist_dict.items() if tgt in test_nodes_set]
            if not dists:
                results.append((np.nan, np.nan, np.nan))
            else:
                dists.sort()
                topk = dists[:self.k] if len(dists) >= self.k else dists
                results.append((dists[0], sum(dists) / len(dists), sum(topk) / len(topk)))
        return results

    def save_or_update_distances(self):
        strategies = [col for col in self.node_df.columns if col not in ["Node", "Explainer", "Dataset", "Model"]]
        existing = pd.read_csv(self.output_path) if os.path.exists(self.output_path) else pd.DataFrame({"Node": self.node_df["Node"]})

        for strategy in strategies:
            if f"{strategy}_min_dist" in existing.columns:
                continue  # already computed

            strategy_nodes = self.node_df[self.node_df[strategy] == 1]["Node"].tolist()
            distances = self._compute_distances_for_nodes(strategy_nodes)
            distance_map = {node: dist for node, dist in zip(strategy_nodes, distances)}

            existing[f"{strategy}_min_dist"] = self.node_df["Node"].map(lambda n: distance_map.get(n, (float("nan"),))[0])
            existing[f"{strategy}_mean_dist"] = self.node_df["Node"].map(lambda n: distance_map.get(n, (None, float("nan")))[1])
            existing[f"{strategy}_top{self.k}_avg_dist"] = self.node_df["Node"].map(lambda n: distance_map.get(n, (None, None, float("nan")))[2])

        existing.to_csv(self.output_path, index=False)
        return existing


    def plot_distance_summary(self, df, save_dir):
        # 指標
        metrics = [
            ("min_dist", "Minimum"),
            ("mean_dist", "Mean"),
            (f"top{self.k}_avg_dist", f"Top-{self.k} Avg")
        ]

        # 所有策略
        strategies = sorted({
            col.rsplit("_", maxsplit=2)[0]
            for col in df.columns
            if any(col.endswith(metric_key) for metric_key, _ in metrics)
        })

        # 準備數據：每個策略對應每個 metric 的值
        data = {metric_label: [] for _, metric_label in metrics}
        valid_strategies = []

        for strategy in strategies:
            has_all = False
            for metric_key, metric_label in metrics:
                col = f"{strategy}_{metric_key}"
                if col in df.columns and pd.notna(df[col]).any():
                    value = df[col].mean()
                    data[metric_label].append(value)
                    has_all = True
                else:
                    data[metric_label].append(np.nan)  # 佔位，方便對齊

            if has_all:
                valid_strategies.append(strategy)

        # 對應 bar group 的 x 軸位置
        x = np.arange(len(valid_strategies))
        width = 0.25

        # 畫圖
        plt.figure(figsize=(max(8, len(valid_strategies) * 1.5), 5))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, (metric_label, color) in enumerate(zip(data.keys(), colors)):
            values = [data[metric_label][j] for j in range(len(strategies)) if strategies[j] in valid_strategies]
            plt.bar(x + i * width, values, width=width, label=metric_label, color=color)

        plt.xticks(x + width, [s.replace("_", " ").title() for s in valid_strategies])
        plt.ylabel("Average Distance")
        plt.title("Distance between Selected Nodes and Test Set")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(save_dir, "distance_summary_by_strategy.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved bar chart to: {save_path}")


# python tools/distance_to_test.py --base_dir stage2_y_edge_0.3 --explainer GNNExplainer --dataset Actor --k 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="stage2_y_edge_0.3", help="Base path to saved results")
    parser.add_argument("--explainer", type=str, default="GNNExplainer", help="Explainer name (e.g., GNNExplainer)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., Actor, Cora)")
    parser.add_argument("--model_name", type=str, default="1_GCN2Classifier", help="Model folder name (e.g., 1_GCN2Classifier)")
    parser.add_argument("--k", type=int, default=3, help="Top-k average distance")
    args = parser.parse_args()

    csv_dir = os.path.join("saved", args.base_dir, args.explainer, args.dataset, args.model_name)

    # Load dataset using your loader
    from data.dataset_loader import GraphDatasetLoader
    loader = GraphDatasetLoader(normalize=False)
    data, _, _ = loader.load_dataset(args.dataset)

    analyzer = NodeDistanceToTestSaver(csv_dir=csv_dir, data=data, k=args.k)
    df = analyzer.save_or_update_distances()
    analyzer.plot_distance_summary(df, csv_dir)

    print(df.head())
