# 目前的random walk 沒有考慮 edge weight 
# 加入特徵邊會移除雙向，還沒驗證
# 還沒加入 self.only_feature_node 的情況
# 因為之後類別型也要全連接，feature value 要當作edge weight 但目前的版本是沒有考慮包含weight的random walk

import torch
import numpy as np
from torch_cluster import random_walk
from collections import defaultdict

# 還沒檢查 但程式跑起來沒問題

import torch
import numpy as np
from collections import defaultdict


class RandomWalkEdgeSelector:
    def __init__(self, data, fraction, selected_nodes, walk_length=10, num_walks=5,
                 feature_type="categorical", device="cpu", manual_nodes=None,
                 top_k_percent_feat=0.1, only_feature_node=False):
        self.data = data
        self.fraction = fraction
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.feature_type = feature_type
        self.device = device
        self.manual_nodes = manual_nodes
        self.top_k_percent_feat = top_k_percent_feat
        self.only_feature_node = only_feature_node
        self.start_nodes = torch.tensor(selected_nodes, dtype=torch.long, device=self.device)

    def run_weighted_random_walk(self):
        edge_index = self.data.edge_index.to(self.device)
        edge_weight = getattr(self.data, 'edge_weight', torch.ones(edge_index.size(1), device=self.device))

        # 預先建立鄰接表與邊權重映射
        row, col = edge_index
        adj_dict = defaultdict(list)
        weight_dict = defaultdict(list)

        for i in range(edge_index.size(1)):
            adj_dict[row[i].item()].append(col[i].item())
            weight_dict[row[i].item()].append(edge_weight[i].item())

        walks = []
        for node in self.start_nodes.repeat_interleave(self.num_walks):
            walk = [node.item()]
            current = node.item()
            for _ in range(self.walk_length):
                neighbors = adj_dict[current]
                if not neighbors:
                    break
                weights = weight_dict[current]
                probs = np.array(weights) / np.sum(weights)
                current = np.random.choice(neighbors, p=probs)
                walk.append(current)
            walks.append(torch.tensor(walk, device=self.device))

        print(f"Generated {len(walks)} weighted walks of length {self.walk_length}")
        return walks

    def get_edge_index_map(self):
        edge_index = self.data.edge_index.to(self.device)
        return {tuple(edge.tolist()): idx for idx, edge in enumerate(edge_index.t())}

    def expand_bidirectional_edges(self, selected_indices, edge_map):
        reverse_edge_map = { (dst, src): idx for (src, dst), idx in edge_map.items() }
        expanded = set(selected_indices)
        for edge, idx in edge_map.items():
            if idx in selected_indices and (edge[1], edge[0]) in reverse_edge_map:
                expanded.add(reverse_edge_map[(edge[1], edge[0])])
        return list(expanded)

    def select_edges(self):
        walks = self.run_weighted_random_walk()
        visited_edge_count = defaultdict(int)

        for walk in walks:
            for i in range(len(walk) - 1):
                src = walk[i].item()
                dst = walk[i + 1].item()
                if src != dst:
                    edge = (min(src, dst), max(src, dst)) if self.data.is_undirected() else (src, dst)
                    visited_edge_count[edge] += 1

        all_edges = list(visited_edge_count.items())
        all_edges.sort(key=lambda x: x[1], reverse=True)

        node_node_mask_np = self.data.node_node_mask.cpu().numpy()
        node_feat_mask_np = self.data.node_feat_mask.cpu().numpy()

        num_ori_edges = node_node_mask_np.sum() if not self.only_feature_node else 0
        num_feat_edges = node_feat_mask_np.sum()
        ori_num_features = self.data.is_feature_node.sum().item()

        edge_map = self.get_edge_index_map()

        selected_ori, selected_feat = [], []
        for edge, _ in all_edges:
            if edge in edge_map:
                idx = edge_map[edge]
                if not self.only_feature_node and node_node_mask_np[idx]:
                    selected_ori.append(idx)
                elif node_feat_mask_np[idx]:
                    selected_feat.append(idx)

        if not self.only_feature_node:
            num_selected_ori = int(num_ori_edges * self.fraction)
            selected_ori = self.expand_bidirectional_edges(selected_ori, edge_map)
            if len(selected_ori) < num_selected_ori:
                remaining_ori = list(set(range(num_ori_edges)) - set(selected_ori))
                needed = num_selected_ori - len(selected_ori)
                extra = torch.tensor(remaining_ori, device=self.device)[torch.randperm(len(remaining_ori))[:needed]].tolist()
                selected_ori += extra
            else:
                selected_ori = selected_ori[:num_selected_ori]
        else:
            selected_ori = []

        if num_feat_edges > 0:
            num_feat_pairs = num_feat_edges // 2
            num_selected_feat_pairs = int(num_feat_pairs * self.top_k_percent_feat)

            feat_edge_base = 0 if self.only_feature_node else num_ori_edges
            selected_feat = np.array(selected_feat)
            rel_idx = selected_feat - feat_edge_base
            pair_idx = rel_idx // 2

            seen_pair_set, selected_pair_ordered = set(), []
            for p in pair_idx:
                if p not in seen_pair_set:
                    seen_pair_set.add(p)
                    selected_pair_ordered.append(p)

            if len(selected_pair_ordered) < num_selected_feat_pairs:
                all_pair_idx = np.arange(num_feat_pairs)
                remaining_pairs = list(set(all_pair_idx) - set(seen_pair_set))
                needed = num_selected_feat_pairs - len(selected_pair_ordered)
                extra = np.random.choice(remaining_pairs, needed, replace=False).tolist()
                selected_pair_ordered += extra
            else:
                selected_pair_ordered = selected_pair_ordered[:num_selected_feat_pairs]

            selected_feat = []
            for p in selected_pair_ordered:
                edge1 = feat_edge_base + 2 * p
                edge2 = edge1 + 1
                selected_feat += [edge1, edge2]
        else:
            selected_feat = []

        selected_indices = selected_ori + selected_feat
        selected_tensor = torch.tensor(selected_indices, dtype=torch.long, device=self.device)

        rel_idx = np.array(selected_feat) - (0 if self.only_feature_node else num_ori_edges)
        pair_idx = rel_idx // 2
        selected_feat_ids = (pair_idx % ori_num_features).tolist()

        ori_edge_visit_ratio = len(selected_ori) / num_ori_edges if num_ori_edges > 0 else 0
        feat_edge_visit_ratio = len(selected_feat) / num_feat_edges if num_feat_edges > 0 else 0

        print(f"[RandomWalkEdgeSelector] Selected {len(selected_ori)} node-node edges and {len(selected_feat)} feature edges.")
        print(f"[RandomWalkEdgeSelector] Edge visit ratio - Node edges: {ori_edge_visit_ratio:.3f}, Feature edges: {feat_edge_visit_ratio:.3f}")

        return selected_tensor, selected_feat_ids, ori_edge_visit_ratio, feat_edge_visit_ratio


        return selected_tensor, selected_feat_ids, ori_edge_visit_ratio, feat_edge_visit_ratio
