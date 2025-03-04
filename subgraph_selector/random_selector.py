import torch
import random

class RandomSubgraphSelector:
    """
    Selects a random subgraph by sampling a fraction of edges 
    from the training subset of the original graph.
    """

    def __init__(self, data, fraction=0.1, seed=None, device="cpu"):
        """
        Initializes the subgraph selector.

        :param data: PyG Data object representing the original graph.
        :param fraction: Fraction of training edges to select for the subgraph (default: 10%).
        :param seed: Random seed for reproducibility (default: None).
        :param device: The device on which tensors should be placed (e.g., 'cpu' or 'cuda').
        """
        self.data = data.to(device)
        self.fraction = fraction
        self.device = device  # 新增這一行，確保裝置統一

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def select_subgraph(self):
        """
        Selects a random subgraph by sampling a fraction of edges 
        only from the training subset.

        :return: selected_edges (Tensor) - Indices of selected edges in the original graph.
        """
        edge_index = self.data.edge_index.to(self.device)  # Move edge index to the correct device
        train_mask = self.data.train_mask.to(self.device)  # Ensure train mask is on device
        train_nodes = torch.where(train_mask)[0]  # Get indices of training nodes

        # Identify edges where both nodes belong to the training set
        mask_train_edges = torch.isin(edge_index[0], train_nodes) & torch.isin(edge_index[1], train_nodes)
        train_edge_indices = torch.where(mask_train_edges)[0]  # Get original edge indices for training edges

        num_train_edges = train_edge_indices.shape[0]
        num_selected = int(num_train_edges * self.fraction)

        if num_selected == 0:
            raise ValueError("No edges selected. Try increasing the fraction.")

        # Randomly sample edges from the training edges
        selected_edges = torch.tensor(random.sample(train_edge_indices.tolist(), num_selected), dtype=torch.long, device=self.device)

        print(f"Selected {num_selected} edges out of {num_train_edges} training edges.")

        return selected_edges  # Returns indices in the original edge_index
