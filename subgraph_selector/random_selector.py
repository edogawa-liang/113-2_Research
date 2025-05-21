import torch
import random

class RandomEdgeSelector:
    """
    Selects a random subgraph by sampling a fraction of edges 
    from the training subset of the original graph. (核心子圖包含整個節點)
    """

    def __init__(self, data, fraction=0.1, seed=None, device="cpu", top_k_percent_feat=0.1):
        """
        Initializes the subgraph selector.

        :param data: PyG Data object representing the original graph.
        :param fraction: Fraction of training edges to select for the subgraph (default: 10%).
        :param seed: Random seed for reproducibility (default: None).
        :param device: The device on which tensors should be placed (e.g., 'cpu' or 'cuda').
        """
        self.data = data.to(device)
        self.fraction = fraction
        self.top_k_percent_feat = top_k_percent_feat
        self.device = device  

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def select_edges(self, num_ori_edges):
        """
        Selects a random subgraph by sampling a fraction of edges 
        only from the training subset.

        :return: selected_edges (Tensor) - Indices of selected edges in the original graph.
        """

        edge_index = self.data.edge_index.to(self.device)
        num_total = edge_index.size(1)
        num_feat = num_total - num_ori_edges

        train_mask = self.data.train_mask.to(self.device)
        train_nodes = torch.where(train_mask)[0]

        # 原始邊: Identify edges where both nodes belong to the training set 
        mask_train_edges = torch.isin(edge_index[0], train_nodes) & torch.isin(edge_index[1], train_nodes)
        train_edge_indices = torch.where(mask_train_edges)[0]  # Get original edge indices for training edges
        num_train_edges = train_edge_indices.shape[0]
        num_selected_ori = int(num_train_edges * self.fraction)

        # 挑選的原始邊
        selected_ori = random.sample(train_edge_indices.tolist(), num_selected_ori)


        # 特徵邊
        if num_feat > 0:
            print("No feature edges found. Selecting from original edges only.")
            feat_edge_indices = list(range(num_ori_edges, num_total))
            num_selected_feat = int(num_feat * self.top_k_percent_feat)
            selected_feat = random.sample(feat_edge_indices, num_selected_feat) if num_selected_feat > 0 else []
        else:
            selected_feat = []
        
        selected_idx = selected_ori + selected_feat

        print(f"Selected {len(selected_ori)} training edges from {num_train_edges} available.")
        print(f"Selected {len(selected_feat)} feature edges from {num_feat} available.")

        return torch.tensor(selected_idx, dtype=torch.long, device=self.device)
