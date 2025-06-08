import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import csv
import numpy as np
import networkx as nx
import torch
from utils.device import DEVICE
from torch_geometric.data import Data
import argparse
from data.dataset_loader import GraphDatasetLoader
from data.feature2node import FeatureNodeConverter
import igraph as ig # 計算betweenness時用到


# def min_max_norm(arr):
#     """Normalize a NumPy array to [0, 1]."""
#     arr = np.array(arr, dtype=float)
#     vmin, vmax = arr.min(), arr.max()
#     if vmax - vmin < 1e-12:
#         return np.zeros_like(arr)
#     return (arr - vmin) / (vmax - vmin)


class NodeImportanceCalculator:
    """
    Compute and save normalized Degree, PageRank, and Betweenness for each node in a PyG Data object.
    """

    def __init__(self, data: Data, dataset_name: str, feature_to_node: bool, only_feature_node: bool, save_dir: str = "saved/node_imp"):
        """
        Initialize with PyG graph and configuration.

        Args:
            data (Data): PyG Data object (already feature-to-node converted if needed).
            dataset_name (str): Dataset name, used for folder/filename.
            feature_to_node (bool): Whether the graph includes feature-to-node edges.
            only_feature_node (bool): Whether only feature-node edges are included (no node-node edges).
            save_dir (str): Directory to save output.
        """
        self.data = data
        self.dataset_name = dataset_name
        self.feature_to_node = feature_to_node
        self.only_feature_node = only_feature_node
        self.save_dir = save_dir

        # 挑出需要使用的邊
        if self.feature_to_node:
            assert hasattr(self.data, "node_node_mask"), "Missing node_node_mask"
            assert hasattr(self.data, "node_feat_mask"), "Missing node_feat_mask"
            if self.only_feature_node:
                mask = self.data.node_feat_mask
            else:
                mask = self.data.node_node_mask | self.data.node_feat_mask
            self.edge_index = self.data.edge_index[:, mask]
            self.edge_weight = self.data.edge_weight[mask] if hasattr(self.data, 'edge_weight') else None
        else:
            # 完全使用原始邊，不套任何 mask
            self.edge_index = self.data.edge_index
            self.edge_weight = self.data.edge_weight if hasattr(self.data, 'edge_weight') else None

        print(f"Using {self.edge_index.size(1)} edges for importance calculation.")



    def compute(self):
        """Convert edge_index to NetworkX graph and compute importance scores."""
        G = nx.Graph()
        edge_index_np = self.edge_index.cpu().numpy()
        edge_weights = (
            self.edge_weight.cpu().numpy()
            if self.edge_weight is not None
            else np.ones(edge_index_np.shape[1], dtype=float)
        )

        # Add weighted edges
        for (u, v), w in zip(edge_index_np.T, edge_weights):
            G.add_edge(int(u), int(v), weight=float(w))

        # Centrality metrics with edge weights
        deg = dict(G.degree(weight="weight"))  # Weighted degree
        print("Complete degree calculation.")
        pr = nx.pagerank(G, alpha=0.85, weight="weight")  # Weighted PageRank
        print("Complete PageRank calculation.")

        # Convert to igraph for faster betweenness, closeness calculation
        print("Converting to igraph...")
        g_ig = ig.Graph()
        g_ig.add_vertices(max(G.nodes) + 1)
        edges = list(G.edges(data="weight"))
        g_ig.add_edges([(u, v) for u, v, w in edges])
        g_ig.es["weight"] = [w for u, v, w in edges]
        print("Complete igraph conversion.")

        # Betweenness with cutoff
        print("Calculating Betweenness with cutoff=4 ...")
        bet_list = g_ig.betweenness(weights="weight", cutoff=4)
        print("Complete Betweenness calculation.")
        bet = {node_id: score for node_id, score in enumerate(bet_list)}

        # Closeness with weights
        clo_list = g_ig.closeness(weights="weight") 
        clo = {node_id: score for node_id, score in enumerate(clo_list)}
        print("Complete Closeness calculation.")


        # Force node list to be ordered: 0 to N-1
        self.nodes = list(range(self.data.num_nodes))
        # self.degree_vals = min_max_norm([deg.get(n, 0.0) for n in self.nodes])
        # self.pagerank_vals = min_max_norm([pr.get(n, 0.0) for n in self.nodes])
        # self.betweenness_vals = min_max_norm([bet.get(n, 0.0) for n in self.nodes])
        # self.closeness_vals = min_max_norm([clo.get(n, 0.0) for n in self.nodes])
        self.degree_vals = [deg.get(n, 0.0) for n in self.nodes]
        self.pagerank_vals = [pr.get(n, 0.0) for n in self.nodes]
        self.betweenness_vals = [bet.get(n, 0.0) for n in self.nodes]
        self.closeness_vals = [clo.get(n, 0.0) for n in self.nodes]




    def save(self):
        """Save metrics to CSV."""
        if self.feature_to_node:
            if self.only_feature_node:
                suffix = f"{self.dataset_name}_fn.csv"
            else:
                suffix = f"{self.dataset_name}_fn_nn.csv"
        else:
            suffix = f"{self.dataset_name}_ori.csv"

        output_dir = os.path.join(self.save_dir, self.dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, suffix)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["node", "degree_norm", "pagerank_norm", "betweenness_norm", "closeness_norm"])
            for i, node in enumerate(self.nodes):
                writer.writerow([
                    node,
                    self.degree_vals[i],
                    self.pagerank_vals[i],
                    self.betweenness_vals[i],
                    self.closeness_vals[i]
                ])
        print(f"Saved node importance to {output_path}")

# python tools/cal_node_imp.py --dataset Cora --feature_to_node --only_feature_node

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate node importance metrics for a graph.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize the dataset.")
    parser.add_argument("--feature_to_node", action="store_true", help="Convert features into nodes.")
    parser.add_argument("--only_feature_node", action="store_true", help="Use only feature-node edges.")

    args = parser.parse_args()

    os.environ['TORCH'] = torch.__version__
    print(f"Using torch version: {torch.__version__}")
    print(f"Using device: {DEVICE}")

    # Load dataset
    loader = GraphDatasetLoader(args.normalize)
    data, num_features, _, feature_type, _ = loader.load_dataset(args.dataset)
    data = data.to(DEVICE)

    # Apply feature-to-node conversion if needed
    if args.feature_to_node:
        converter = FeatureNodeConverter(feature_type=feature_type, device=DEVICE)
        data = converter.convert(data)

    # Run importance calculation
    calculator = NodeImportanceCalculator(
        data=data,
        dataset_name=args.dataset,
        feature_to_node=args.feature_to_node,
        only_feature_node=args.only_feature_node,
    )
    calculator.compute()
    calculator.save()
