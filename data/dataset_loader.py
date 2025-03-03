import os
import torch
from torch_geometric.datasets import FacebookPagePage, GitHub
import torch_geometric.transforms as T

class GraphDatasetLoader:
    """
    A class for loading graph datasets using PyTorch Geometric.
    """
    def __init__(self):
        """
        Initializes the dataset loader, sets device, and defines transformations.
        """
        os.environ['TORCH'] = torch.__version__
        print(f"Using torch version: {torch.__version__}")
        
        # Set device
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print("==============================================================\n")
        
        torch.manual_seed(42)
        self.transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(self.device),  
            T.RandomNodeSplit(num_val=0.1, num_test=0.1)
        ])
                
        # Available datasets
        self.datasets = {
            "FacebookPagePage": lambda: FacebookPagePage(root='/tmp/FacebookPagePage', transform=self.transform),
            "GitHub": lambda: GitHub(root='/tmp/GitHub', transform=self.transform),
        }
    
    def load_dataset(self, name: str):
        """
        Loads the dataset by name.
        
        :param name: Name of the dataset to load.
        :return: Tuple containing the dataset and its first graph data object.
        """
        if name not in self.datasets:
            raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(self.datasets.keys())}")
        
        dataset = self.datasets[name]()  # Load dataset
        print(f"Dataset: {dataset}:")
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of features: {dataset.num_features}")
        print(f"Number of classes: {dataset.num_classes}")
        
        data = dataset[0]  # Retrieve first graph data
        print("Graph data:")
        print(data)
        print("Dataset loaded successfully.")
        print("==============================================================\n")
        
        # Display graph statistics
        # print(f"Number of nodes: {data.num_nodes}")
        # print(f"Labeled Nodes: {(data.y >= 0).sum().item()}")  # Number of labeled nodes
        # print(f"Number of edges: {data.num_edges}")
        # print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
        # print(f"Number of training nodes: {data.train_mask.sum()}")
        # print(f"Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}")
        # print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        # print(f"Has self-loops: {data.has_self_loops()}")
        # print(f"Is undirected: {data.is_undirected()}")
        # print(f"Node Feature: {data.x}")
        
        return data, dataset.num_features, dataset.num_classes

if __name__ == "__main__":
    loader = GraphDatasetLoader()
    dataset_name = input(f"Enter dataset name {list(loader.datasets.keys())}: ")
    data = loader.load_dataset(dataset_name)
