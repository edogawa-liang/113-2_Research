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

