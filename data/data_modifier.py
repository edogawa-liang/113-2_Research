import torch
from torch_geometric.data import Data
import copy

class GraphModifier:
    """
    A utility to modify graph data by setting a specified node feature column as y 
    and removing it from the node features.
    """

    def __init__(self, data):
        """
        Initializes the GraphModifier with a given PyG Data object.
        """
        self.original_data = data 
        self.num_features = data.x.shape[1]  

    def modify_graph(self, feature_indices):
        """
        Iterates over the given feature indices, treating each as y and removing it from x.
        """
        modified_graphs = []

        for idx in feature_indices:
            if idx >= self.num_features:
                raise ValueError(f"Feature index {idx} is out of bounds for {self.num_features} features.")

            # Deep copy original data to avoid modification
            new_data = copy.deepcopy(self.original_data)

            # Set y as the feature column and remove it from x
            new_data.y = new_data.x[:, idx].clone()
            new_data.x = torch.cat((new_data.x[:, :idx], new_data.x[:, idx+1:]), dim=1)

            print(f"Modified Graph: Feature {idx} set as y and removed from x. New x shape: {new_data.x.shape}")
            modified_graphs.append(new_data)

        return modified_graphs



# Example Usage
if __name__ == "__main__":
    # Create a sample graph with 5 nodes and 4 features
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0],
                      [13.0, 14.0, 15.0, 16.0],
                      [17.0, 18.0, 19.0, 20.0]])
    
    edge_index = torch.tensor([[0, 1, 2, 3],
                               [1, 2, 3, 4]])  # Simple edge connections

    data = Data(x=x, edge_index=edge_index)

    # Specify feature indices to use as y
    feature_indices = [0, 2]  # Remove feature 0 and 2

    modifier = GraphModifier(data)
    modified_graphs = modifier.modify_graph(feature_indices)

    for i, graph in enumerate(modified_graphs):
        print(f"\nGraph {i+1}: y = Feature {feature_indices[i]}")
        print("New x:", graph.x)
        print("y:", graph.y)
