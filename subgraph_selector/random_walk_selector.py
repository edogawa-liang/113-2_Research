import torch
import numpy as np
from torch_cluster import random_walk
from collections import defaultdict


# 目前的random walk 沒有考慮 edge weight
class RandomWalkEdgeSelector:
    """
    Selects the top fraction of most frequently visited edges (by random walk) for removal.
    """

    def __init__(self, data, fraction, selected_nodes, walk_length=10, num_walks=5, 
                 feature_type="categorical", device="cpu", manual_nodes=None, top_k_percent_feat=0.1):
        """
        :param data: PyG graph data
        :param fraction: Fraction of most frequently visited edges to remove (0.1 → top 10%)
        :param walk_length: Number of steps per random walk
        :param num_walks: Number of random walks per starting node
        :param node_choose: Strategy for selecting starting nodes ("random", "high_degree", "top_pagerank", "manual")
        :param device: Device to use (e.g., 'cpu' or 'cuda')
        :param manual_nodes: Manually specified nodes if node_choose is "manual"
        """
        self.data = data
        self.fraction = fraction
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.feature_type = feature_type  # 用於選擇特徵邊
        self.device = device
        self.manual_nodes = manual_nodes
        self.top_k_percent_feat = top_k_percent_feat
        self.start_nodes = torch.tensor(selected_nodes, dtype=torch.long, device=self.device)


       
    def select_edges(self):
        """
        1. Perform random walks using PyG's `random_walk()`.
        2. Count how many times each edge is visited.
        3. Remove the top `self.fraction` fraction of edges (by frequency).
        4. Return indices of these edges in the original `edge_index`.
        """
        edge_index = self.data.edge_index.to(self.device)

        # 檢查圖是否為無向圖
        is_undirected = self.data.is_undirected()
        print(f"Graph is {'Undirected' if is_undirected else 'Directed'}.")

        # 執行 random_walk
        walks = random_walk(
            edge_index[0], 
            edge_index[1], 
            start=self.start_nodes.repeat_interleave(self.num_walks), 
            walk_length=self.walk_length
        )
        print(f"Shape of walks: {walks.shape}")

        # 統計每條邊被訪問的次數
        visited_edge_count = defaultdict(int)

        # 走訪結果可能是 2D tensor (num_walks_total, walk_length+1)
        for walk in walks:
            # walk 可能是 1D
            # 先確保它是 list or 1D
            if walk.dim() == 0:
                # 單個 scalar，無法形成路徑
                continue
            # walk 是 1D tensor，表示一條完整的 random walk
            for i in range(len(walk) - 1):
                src = walk[i].item()
                dst = walk[i+1].item()
                if src != dst:  # 避免 self-loop
                    if is_undirected:
                        edge = (min(src, dst), max(src, dst))
                    else:
                        edge = (src, dst)
                    visited_edge_count[edge] += 1


        # 根據出現次數排序，取前 self.fraction 比例
        all_edges = list(visited_edge_count.items())  # [(edge, freq), ...]
        all_edges.sort(key=lambda x: x[1], reverse=True)

        # Edge masks
        node_node_mask_np = self.data.node_node_mask.cpu().numpy()
        node_feat_mask_np = self.data.node_feat_mask.cpu().numpy()
        
        num_ori_edges = node_node_mask_np.sum()  # 幾條 node-node edge
        num_feat_edges = node_feat_mask_np.sum() # 幾條 node-feature edge
        num_total_edges = len(node_node_mask_np)
        ori_num_features = self.data.is_feature_node.sum().item()  # 原始特徵節點數量


        # 建立 (src, dst) → edge_index 映射
        edge_map = {
            tuple(edge.tolist()): idx for idx, edge in enumerate(edge_index.t())
        }

        selected_ori = []
        selected_feat = []

        for edge, _ in all_edges:
            if edge in edge_map:
                idx = edge_map[edge]
                if node_node_mask_np[idx]:
                    selected_ori.append(idx)
                elif node_feat_mask_np[idx]:
                    selected_feat.append(idx)
                else:
                    print(f"[Warning] Edge idx={idx} 不在 node_node_mask 或 node_feat_mask 裡！")


        num_visited_ori_edges = len(selected_ori)
        num_visited_feat_edges = len(selected_feat)

        print(f"走過的原始邊數量: {num_visited_ori_edges}")
        print(f"走過的特徵邊數量: {num_visited_feat_edges}")
        
        # 按比例選擇
        num_selected_ori = int(num_ori_edges * self.fraction)
        # selected_ori = selected_ori[:num_selected_ori]
        print(f"應挑 {num_selected_ori} original edges (fraction {self.fraction})")
        print(f"實際走的原始邊比例: {num_visited_ori_edges / num_ori_edges * 100:.2f}%")
        
        # 如果選擇的原始邊數量少於應挑的數量，則從剩餘的原始邊中補齊
        if len(selected_ori) < num_selected_ori:
            print(f" 探索到的原始邊只有 {len(selected_ori)}，少於應挑的 {num_selected_ori}，將從其他原始邊補齊。")
            # 找出還沒被走過的原始邊 index
            remaining_ori = list(set(range(num_ori_edges)) - set(selected_ori))
            needed_ori = num_selected_ori - len(selected_ori)
            if len(remaining_ori) >= needed_ori:
                additional_ori = torch.tensor(remaining_ori, device=self.device)[torch.randperm(len(remaining_ori))[:needed_ori]].tolist()
                selected_ori += additional_ori
            else:
                print(f"可補充的原始邊不足，只補上 {len(remaining_ori)} 條。")
                selected_ori += remaining_ori
        else:
            selected_ori = selected_ori[:num_selected_ori]


        if num_feat_edges > 0:
            if self.feature_type == "categorical":
                num_selected_feat = int(num_feat_edges * self.top_k_percent_feat) # 只有有值(1)才會被納入特徵邊候選
            elif self.feature_type == "continuous":
                num_selected_feat = int(num_feat_edges * self.top_k_percent_feat)

            print(f"應挑 {num_selected_feat} feature edges (top {self.top_k_percent_feat * 100}%)")
            print(f"實際走的特徵邊比例: {num_visited_feat_edges / num_feat_edges * 100:.2f}%")

            # 如果選擇的特徵邊數量少於應挑的數量，則從剩餘的特徵邊中補齊
            if len(selected_feat) < num_selected_feat:
                print(f"探索到的特徵邊只有 {len(selected_feat)}，少於應挑的 {num_selected_feat}，將從其他特徵邊補齊。")
                remaining_feat = list(range(num_ori_edges, num_total_edges))
                remaining_feat = list(set(remaining_feat) - set(selected_feat))
                needed_feat = num_selected_feat - len(selected_feat)
                if len(remaining_feat) >= needed_feat:
                    additional_feat = torch.tensor(remaining_feat, device=self.device)[torch.randperm(len(remaining_feat))[:needed_feat]].tolist()
                    selected_feat += additional_feat
                else:
                    print(f"可補充的特徵邊不足，只補上 {len(remaining_feat)} 條。")
                    selected_feat += remaining_feat
            else:
                selected_feat = selected_feat[:num_selected_feat]

        else:
            selected_feat = []
            print("No feature edges found.")

        selected_indices = selected_ori + selected_feat
        selected_tensor = torch.tensor(selected_indices, dtype=torch.long, device=self.device)

        ori_edge_visit_ratio = num_visited_ori_edges / num_ori_edges if num_ori_edges > 0 else 0
        feat_edge_visit_ratio = num_visited_feat_edges / num_feat_edges if num_feat_edges > 0 else 0

        selected_feat = np.array(selected_feat)
        rel_idx = selected_feat - num_ori_edges
        pair_idx = rel_idx // 2
        selected_feat_ids = (pair_idx % ori_num_features).tolist()

        return selected_tensor, selected_feat_ids, ori_edge_visit_ratio, feat_edge_visit_ratio
