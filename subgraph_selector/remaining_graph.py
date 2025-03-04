import torch

class RemainingGraphConstructor:
    """
    A class for constructing the remaining graph after removing a selected subgraph.
    """

    def __init__(self, data, selected_edges):
        """
        Initializes the remaining graph constructor.

        :param data: PyG Data object representing the original graph.
        :param selected_edges: List of edge indices that were removed.
        """
        self.data = data
        self.selected_edges = torch.tensor(selected_edges, dtype=torch.long)  # Convert to tensor for efficient indexing

    def get_remaining_graph(self):
        """
        Constructs the remaining graph after removing selected edges.

        :return: A PyG Data object representing the remaining graph.
        """
        edge_index = self.data.edge_index
        num_edges = edge_index.shape[1]

        # Create a mask for remaining edges (those NOT in selected_edges)
        mask = ~torch.isin(torch.arange(num_edges, device=edge_index.device), self.selected_edges)

        # Get remaining edges
        remaining_edge_index = edge_index[:, mask]

        # Clone the original data and update only edge_index
        remaining_graph = self.data.clone()
        remaining_graph.edge_index = remaining_edge_index

        print(f"Remaining graph has {remaining_graph.num_nodes} nodes and {remaining_graph.edge_index.shape[1]} edges.")
        
        # index 都沒有改變，但可能留下孤立節點。
        # 孤立節點一定是 training node，因為只移除有training edge。
        return remaining_graph
