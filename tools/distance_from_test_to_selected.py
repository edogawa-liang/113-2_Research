# 測試節點到被選中節點的距離。
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

class TestToNodeDistanceSaver:
    def __init__(self, csv_dir, data, k=3):
        self.csv_dir = csv_dir
        self.record_path = os.path.join(csv_dir, "node_record.csv")
        self.output_path = os.path.join(csv_dir, "distance_from_test_to_selected.csv")
        self.data = data.clone()
        self.k = k

        self.test_nodes = self.data.test_mask.nonzero(as_tuple=True)[0].tolist()
        self.graph = to_networkx(self.data, to_undirected=True)
        self.node_df = pd.read_csv(self.record_path)
        print(f"Read the nodes information from: {self.record_path}")

    def _compute_distances_from_test_to_nodes(self, target_nodes):
        target_nodes_set = set(target_nodes)
        results = []

        for test_node in self.test_nodes:
            if test_node not in self.graph:
                results.append((np.nan, np.nan, np.nan))
                continue

            dist_dict = nx.single_source_shortest_path_length(self.graph, source=test_node)
            dists = [dist for tgt, dist in dist_dict.items() if tgt in target_nodes_set]

            if not dists:
                results.append((np.nan, np.nan, np.nan))
            else:
                dists.sort()
                topk = dists[:self.k] if len(dists) >= self.k else dists
                results.append((dists[0], sum(dists) / len(dists), sum(topk) / len(topk)))

        return results

    def save_or_update_distances(self):
        strategies = [col for col in self.node_df.columns if col not in ["Node", "Explainer", "Dataset", "Model"]]
        existing = pd.DataFrame({"TestNode": self.test_nodes})

        for strategy in strategies:
            selected_nodes = self.node_df[self.node_df[strategy] == 1]["Node"].tolist()
            distances = self._compute_distances_from_test_to_nodes(selected_nodes)
            distance_map = {node: dist for node, dist in zip(self.test_nodes, distances)}

            existing[f"{strategy}_min_dist"] = existing["TestNode"].map(lambda n: distance_map.get(n, (float("nan"),))[0])
            existing[f"{strategy}_mean_dist"] = existing["TestNode"].map(lambda n: distance_map.get(n, (None, float("nan")))[1])
            existing[f"{strategy}_top{self.k}_avg_dist"] = existing["TestNode"].map(lambda n: distance_map.get(n, (None, None, float("nan")))[2])

        existing.to_csv(self.output_path, index=False)
        print(f"Saved distance file: {self.output_path}")
        return existing

    def plot_distance_summary(self, df, save_dir):
        import matplotlib.pyplot as plt
        import numpy as np

        metrics = [
            ("min_dist", "Minimum"),
            ("mean_dist", "Mean"),
            (f"top{self.k}_avg_dist", f"Top-{self.k} Avg")
        ]

        # 正確找出所有策略（去掉 metric 結尾）
        strategies = sorted({
            col[: -len(metric_key) - 1]  # 去掉 "_mean_dist" 之類尾巴
            for col in df.columns
            for metric_key, _ in metrics
            if col.endswith(metric_key)
        })

        data = {s: [] for s in strategies}
        labels = [label for _, label in metrics]

        for s in strategies:
            for metric_key, _ in metrics:
                col = f"{s}_{metric_key}"
                if col in df.columns and pd.notna(df[col]).any():
                    val = df[col].mean()
                else:
                    val = np.nan
                data[s].append(val)

        x = np.arange(len(labels))
        width = 0.8 / len(strategies)
        colors = plt.cm.tab10.colors

        plt.figure(figsize=(max(8, len(labels) * 2.5), 5))

        for i, s in enumerate(strategies):
            values = data[s]
            plt.bar(x + i * width, values, width=width, label=s.replace("_", " ").title(), color=colors[i % len(colors)])

        plt.xticks(x + width * (len(strategies) - 1) / 2, labels)
        plt.ylabel("Average Distance")
        plt.title("Distance from Test Nodes to Selected Nodes")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(save_dir, "distance_from_test_to_selected.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved grouped bar chart: {save_path}")


# python tools/distance_from_test_to_selected.py --base_dir stage2_y_edge_0.3 --dataset Actor --k 3
# explainer, model_name 的不同跟選取的節點沒有關係。

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="stage2_y_edge_0.3", help="Base path to saved results")
    parser.add_argument("--explainer", type=str, default="GNNExplainer", help="Explainer name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model_name", type=str, default="1_GCN2Classifier", help="Model folder name")
    parser.add_argument("--k", type=int, default=3, help="Top-k average distance")
    args = parser.parse_args()

    csv_dir = os.path.join("saved", args.base_dir, args.explainer, args.dataset, args.model_name)

    loader = GraphDatasetLoader(normalize=False)
    data, _, _ = loader.load_dataset(args.dataset)

    analyzer = TestToNodeDistanceSaver(csv_dir=csv_dir, data=data, k=args.k)
    df = analyzer.save_or_update_distances()
    analyzer.plot_distance_summary(df, csv_dir)
