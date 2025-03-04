import torch

class RemainingGraphConstructor:
    """
    A class for constructing the remaining graph after removing a selected subgraph.
    """

    def __init__(self, data, selected_edges, device="cpu"):
        """
        Initializes the remaining graph constructor.

        :param data: PyG Data object representing the original graph.
        :param selected_edges: Tensor of edge indices that were removed.
        :param device: The device on which tensors should be placed.
        """
        self.data = data.to(device)
        self.selected_edges = selected_edges.to(device)  # 確保 selected_edges 也在 device 上
        self.device = device

    def get_remaining_graph(self):
        """
        Constructs the remaining graph after removing selected edges.

        :return: A PyG Data object representing the remaining graph.
        """
        edge_index = self.data.edge_index.to(self.device)
        num_edges = edge_index.shape[1]

        # 確保所有 tensor 在相同的 device 上
        mask = ~torch.isin(torch.arange(num_edges, device=self.device), self.selected_edges)

        # Get remaining edges
        remaining_edge_index = edge_index[:, mask]

        # Clone the original data and update only edge_index
        remaining_graph = self.data.clone()
        remaining_graph.edge_index = remaining_edge_index

        print(f"Remaining graph has {remaining_graph.num_nodes} nodes and {remaining_graph.edge_index.shape[1]} edges.")
        # index 都沒有改變，但可能留下孤立節點。
        # 孤立節點一定是 training node，因為只移除有training edge。
        return remaining_graph
