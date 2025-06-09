# pick_node.py

import torch
import pandas as pd
import numpy as np
import os
from torch_geometric.utils import k_hop_subgraph

class NodePicker:
    def __init__(self, data, dataset_name, node_choose, feature_to_node=False, only_feature_node=False, imp_dir="saved/node_imp"):
        self.data = data
        self.dataset_name = dataset_name
        self.node_choose = node_choose
        self.feature_to_node = feature_to_node
        self.only_feature_node = only_feature_node
        self.imp_dir = imp_dir

        # 只看原始 node
        is_original_node = data.is_original_node.cpu().numpy()
        self.original_node_indices = np.where(is_original_node)[0]

        # 只看原始 node-node edge
        node_node_mask = data.node_node_mask.cpu().numpy()
        self.num_ori_edges = node_node_mask.sum()

        # 原始 node 數量 N
        self.num_ori_nodes = len(self.original_node_indices)
        print(f"Original nodes: {self.num_ori_nodes}, Original edges: {self.num_ori_edges}")

    def load_importance(self):
        if self.feature_to_node:
            if self.only_feature_node:
                suffix = f"{self.dataset_name}_fn.csv"
            else:
                suffix = f"{self.dataset_name}_fn_nn.csv"
        else:
            suffix = f"{self.dataset_name}_ori.csv"

        imp_path = os.path.join(self.imp_dir, self.dataset_name, suffix)
        print(f"Loading importance from {imp_path}")
        df = pd.read_csv(imp_path)

        # 應該有 columns: node_id, Degree, PageRank, Betweenness, Closeness
        return df

    def pick_nodes(self):
        train_mask = self.data.train_mask.cpu().numpy()
        train_node_indices = np.where(train_mask)[0]

        # 只挑原始 node + train mask
        train_original_node_indices = np.intersect1d(train_node_indices, self.original_node_indices)

        print(f"Original train nodes: {len(train_original_node_indices)} / {len(train_node_indices)}")

        if self.node_choose == "all_train":
            selected_nodes = train_original_node_indices.tolist()
            return selected_nodes

        # Load importance
        df = self.load_importance()
        df = df[df['node_id'].isin(train_original_node_indices)]

        # Sort
        if self.node_choose == "random":
            selected_nodes = df['node_id'].sample(frac=1, random_state=42).tolist()
        else:
            assert self.node_choose in ["Degree", "PageRank", "Betweenness", "Closeness"], "Unknown node_choose!"
            selected_nodes = df.sort_values(by=self.node_choose, ascending=False)['node_id'].tolist()

        # 挑 N 個
        # N = E * r / d^2  (近似版) → 這裡先簡單抓 10% node 你可以改公式
        target_ratio = 0.1
        N = int(self.num_ori_nodes * target_ratio)
        print(f"Picking top {N} nodes ({target_ratio*100:.1f}%)")

        selected_nodes = selected_nodes[:N]
        return selected_nodes

    def compute_coverage(self, selected_nodes):
        # 節點比例
        node_ratio = len(selected_nodes) / self.num_ori_nodes

        # 2-hop 內鄰居
        edge_index = self.data.edge_index.cpu()
        selected_nodes_tensor = torch.tensor(selected_nodes, dtype=torch.long)

        subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
            selected_nodes_tensor, num_hops=2, edge_index=edge_index, relabel_nodes=False
        )

        subset_np = subset.cpu().numpy()
        edge_mask_np = edge_mask.cpu().numpy()

        # 節點 coverage
        num_2hop_nodes = len(subset_np)
        node_2hop_ratio = num_2hop_nodes / self.num_ori_nodes

        # 邊 coverage
        num_2hop_edges = edge_mask_np.sum()
        edge_2hop_ratio = num_2hop_edges / self.num_ori_edges

        print(f"Node ratio: {node_ratio:.4f}, 2-hop node ratio: {node_2hop_ratio:.4f}, 2-hop edge ratio: {edge_2hop_ratio:.4f}")

        return node_ratio, node_2hop_ratio, edge_2hop_ratio
