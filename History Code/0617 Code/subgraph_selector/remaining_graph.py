import torch

class RemainingGraphConstructor:
    """
    A class for constructing the remaining graph after removing a selected subgraph and optionally removing node features.
    """

    def __init__(self, data, selected_edges, selected_feat_mask=None, device="cpu"):
        """
        Initializes the remaining graph constructor.

        :param data: PyG Data object representing the original graph.
        :param selected_edges: Tensor of edge indices that were removed.
        :param selected_feat_mask: Optional [num_nodes, num_features] mask tensor indicating retained features (1) and removed (0).
        :param device: The device on which tensors should be placed.
        """
        self.data = data.to(device)
        self.selected_edges = selected_edges.to(device)
        self.selected_feat_mask = selected_feat_mask.to(device) if selected_feat_mask is not None else None
        self.device = device

    # 在核心子圖內的邊和點會用1注記 所以要留下0的部分
    def get_remaining_graph(self):
        """
        Constructs the remaining graph after removing selected edges and masking features.

        :return: A PyG Data object representing the remaining graph.
        """
        edge_index = self.data.edge_index.to(self.device)
        num_edges = edge_index.shape[1]

        # 1. Remove selected edges
        mask = ~torch.isin(torch.arange(num_edges, device=self.device), self.selected_edges)
        remaining_edge_index = edge_index[:, mask]

        # 2. Clone and update edge_index
        remaining_graph = self.data.clone()
        remaining_graph.edge_index = remaining_edge_index

        # 👉 新增：處理 edge_weight
        if hasattr(self.data, 'edge_weight') and self.data.edge_weight is not None:
            edge_weight = self.data.edge_weight.to(self.device)
            remaining_graph.edge_weight = edge_weight[mask]

        # 3. Optionally remove (mask) node features
        if self.selected_feat_mask is not None:
            if self.selected_feat_mask.shape != self.data.x.shape:
                raise ValueError(f"Feature mask shape {self.selected_feat_mask.shape} does not match data.x shape {self.data.x.shape}")
            
            # 注意這裡是 1 表示要移除 → 所以要反過來
            remaining_graph.x = self.data.x * (1 - self.selected_feat_mask)
            print("Applied inverted feature mask to remove selected features.")

        print(f"Remaining graph has {remaining_graph.num_nodes} nodes and {remaining_graph.edge_index.shape[1]} edges.")
        return remaining_graph

