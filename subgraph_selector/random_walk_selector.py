import torch
from torch_cluster import random_walk
from collections import defaultdict

from subgraph_selector.utils.choose_node import ChooseNodeSelector

class RandomWalkEdgeSelector:
    """
    Selects the top fraction of most frequently visited edges (by random walk) for removal.
    """

    def __init__(self, data, node_ratio="auto", edge_ratio=0.5, fraction=0.1,  
                 walk_length=10, num_walks=5, node_choose="top_pagerank", device="cpu", manual_nodes=None, mask_type="train"):
        """
        :param data: PyG graph data
        :param node_ratio: "auto" for automatic calculation or a numeric value to manually set node selection ratio
        :param edge_ratio: Ensures sufficient edges in the subgraph, required only if node_ratio is 'auto'
        :param fraction: Fraction of most frequently visited edges to remove (0.1 → top 10%)
        :param walk_length: Number of steps per random walk
        :param num_walks: Number of random walks per starting node
        :param node_choose: Strategy for selecting starting nodes ("random", "high_degree", "top_pagerank", "manual")
        :param device: Device to use (e.g., 'cpu' or 'cuda')
        :param manual_nodes: Manually specified nodes if node_choose is "manual"
        """
        self.data = data
        self.node_ratio = node_ratio
        self.edge_ratio = edge_ratio
        self.fraction = fraction
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.node_choose = node_choose
        self.device = device
        self.manual_nodes = manual_nodes
        self.mask_type = mask_type

        self._get_start_nodes()
        self._calculate_neighbor_edges()


    def _get_start_nodes(self):
        """
        Uses `ChooseNodeSelector` to select the starting nodes based on the given strategy.
        """
        node_selector = ChooseNodeSelector(self.data, node_ratio=self.node_ratio, edge_ratio=self.edge_ratio,
                                           strategy=self.node_choose, manual_nodes=self.manual_nodes, mask_type=self.mask_type)
        
        selected_nodes = node_selector.select_nodes()
        self.start_nodes = torch.tensor(selected_nodes, dtype=torch.long, device=self.device)
        self.final_node_ratio = len(selected_nodes) / self.data.x.shape[0]

    def _calculate_neighbor_edges(self):
        """
        計算選中的節點的鄰居邊數量。
        """
        edge_index = self.data.edge_index
        selected_set = set(self.start_nodes.tolist())
        self.neighbor_edges = [
            (src, dst) for src, dst in edge_index.t().tolist() if src in selected_set or dst in selected_set
        ]
        self.neighbor_edge_ratio = len(self.neighbor_edges)/self.data.edge_index.shape[1]

    def get_final_node_ratio(self):
        return self.final_node_ratio
    
    def get_neighbor_edge_ratio(self):
        return self.neighbor_edge_ratio
       

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
        # top_k = int(len(all_edges) * self.fraction)
        top_k = int(self.data.edge_index.shape[1] * self.fraction)
        print(top_k)
        top_visited_edges = [edge for edge, freq in all_edges[:top_k]]

        # 對照 original edge_index 的映射
        train_edges = {
            tuple(edge.tolist()): idx for idx, edge in enumerate(edge_index.t())
        }  # (src, dst) → index

        # 取得最常被走過的邊之索引
        edge_indices_to_remove = [
            train_edges[e] for e in top_visited_edges if e in train_edges
        ]

        print(f"Selected {len(edge_indices_to_remove)} edges out of {edge_index.shape[1]} total edges (top {self.fraction*100}%).")

        # 回傳要移除的邊索引
        return torch.tensor(edge_indices_to_remove, dtype=torch.long, device=self.device).flatten()
