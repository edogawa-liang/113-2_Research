import torch
import random
import numpy as np


# 會用 feature selector 的都是沒有 feature to node 的
class RandomFeatureSelector:
    def __init__(self, num_nodes, num_features, top_k_percent_feat=0.1, same_feat=True, seed=123, device="cpu"):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.top_k_percent_feat = top_k_percent_feat
        self.same_feat = same_feat
        self.device = device

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def select_node_features(self):
        k = int(self.num_features * self.top_k_percent_feat)
        mask = torch.zeros((self.num_nodes, self.num_features), dtype=torch.float32, device=self.device)

        if self.same_feat:
            selected_feat = random.sample(range(self.num_features), k)
            mask[:, selected_feat] = 1.0
            print(f"[same_feat=True] All nodes select same {k} features.")
        else:
            for i in range(self.num_nodes):
                selected_feats = random.sample(range(self.num_features), k)
                mask[i, selected_feats] = 1.0
            print(f"[same_feat=False] Each node selects {k} features independently.")

        return mask  # shape: [num_nodes, num_features]


class RandomEdgeSelector:
    # feature_to_node 有問題 還沒修(但不會使用到)
    def __init__(self, data, fraction=0.1, seed=123, device="cpu", top_k_percent_feat=0.1, feature_to_node=None):
        self.data = data.to(device)
        self.fraction = fraction
        self.top_k_percent_feat = top_k_percent_feat # feature_to_node 時才需要用到
        self.device = device
        self.feature_to_node = feature_to_node

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def select_edges(self, num_ori_edges):
        """
        Selects a random subgraph:
        - From training nodes, randomly select a fraction of node-node edges.
        - From all node-feature edges, randomly select top_k_percent_feat * num_nodes edges.

        :param num_ori_edges: number of original node-node edges in the graph
        :return: selected_edges (Tensor): indices of selected edges in the edge_index
        """
        edge_index = self.data.edge_index.to(self.device)
        num_total = edge_index.size(1)
        num_feat = num_total - num_ori_edges
        num_nodes = self.data.num_nodes
        num_features = self.data.x.size(1)

        # Combine train/val/unknown masks → valid nodes
        combined_mask = (
            self.data.train_mask.to(self.device)
            | self.data.val_mask.to(self.device)
            | self.data.unknown_mask.to(self.device)
        )
        valid_nodes = torch.where(combined_mask)[0]

        # ========== 篩選訓練節點間的 node-node 邊 ==========
        ori_edge_index = edge_index[:, :num_ori_edges]
        mask_valid_edges = torch.isin(ori_edge_index[0], valid_nodes) & torch.isin(ori_edge_index[1], valid_nodes)
        valid_edge_indices = torch.where(mask_valid_edges)[0]

        num_valid_edges = valid_edge_indices.shape[0]
        num_selected_ori = int(num_valid_edges * self.fraction)
        selected_ori = random.sample(valid_edge_indices.tolist(), num_selected_ori)

        # ========== 篩選 node-feature 邊 ==========
        if num_feat > 0:
            print("Found feature edges. Selecting from feature edges.")
            node_feat_indices = list(range(num_ori_edges, num_total))
            num_selected_feat = int((num_nodes - num_features) * self.top_k_percent_feat)
            selected_feat = random.sample(node_feat_indices, min(num_selected_feat, len(node_feat_indices)))
        else:
            selected_feat = []

        # ========== Combine ==========
        selected_idx = selected_ori + selected_feat

        print(f"Selected {len(selected_ori)} node-node edges from {num_valid_edges} available (train + val + unknown).")
        print(f"Selected {len(selected_feat)} node-feature edges from {num_feat} available.")

        return torch.tensor(selected_idx, dtype=torch.long, device=self.device)
