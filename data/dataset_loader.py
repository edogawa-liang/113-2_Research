import torch
from torch_geometric.datasets import FacebookPagePage, GitHub, Planetoid, KarateClub, Amazon, Airports, Actor
import torch_geometric.transforms as T

class GraphDatasetLoader:
    """
    A class for loading graph datasets using PyTorch Geometric.
    """
    def __init__(self, normalize=False):
        """
        Initializes the dataset loader, sets device, and defines transformations.
        """

        torch.manual_seed(42)

        transforms = []
        if normalize:
            transforms.append(T.NormalizeFeatures())  # 只有 normalize=True 才添加
        # transforms.append(T.RandomNodeSplit(num_val=0.1, num_test=0.1))

        self.transform = T.Compose(transforms)
                
        # Available datasets
        self.datasets = {
            "FacebookPagePage": lambda: FacebookPagePage(root='/tmp/FacebookPagePage', transform=self.transform),
            "GitHub": lambda: GitHub(root='/tmp/GitHub', transform=self.transform),
            "Cora": lambda: Planetoid(root='/tmp/Cora', name='Cora', transform=self.transform),
            "CiteSeer": lambda: Planetoid(root='/tmp/CiteSeer', name='CiteSeer', transform=self.transform),
            "PubMed": lambda: Planetoid(root='/tmp/PubMed', name='PubMed', transform=self.transform),
            "Amazon": lambda: Amazon(root='/tmp/Amazon', name='Computers', transform=self.transform),
            "Airports": lambda: Airports(root='/tmp/Airports', name='USA', transform=self.transform),
            "Actor": lambda: Actor(root='/tmp/Actor', transform=self.transform),
        }
        self.feature_type_map = {
            "Cora": "categorical",      # One-hot encoding of paper keywords (binary vector, each dimension = 1 keyword present)
            "CiteSeer": "categorical",  # One-hot encoding of paper keywords (binary vector, each dimension = 1 keyword present)
            "PubMed": "continuous",     # TF-IDF feature vector of paper words (float values per dimension)
            "FacebookPagePage": "continuous",  # Page feature embeddings (continuous floats, typically from page activity / metadata)
            "GitHub": "continuous",     # Node embeddings of GitHub users/repos (continuous floats, e.g. learned or handcrafted embeddings)
            "Amazon": "categorical",    # Bag-of-words vector of product reviews/descriptions (binary vector, 0/1 per word)
            "Airports": "categorical",  # One-hot encoding of node type (one-hot vector per node type, e.g. hub / regional / small / intl airport type)
            "Actor": "categorical"      # Multi-hot encoding of keywords from movies the actor participated in (binary vector, 0/1 per keyword)
        }

    
    def load_dataset(self, name: str):
        """
        Loads the dataset by name.
        
        :param name: Name of the dataset to load.
        :return: Tuple containing the dataset and its first graph data object.
        """
        if name not in self.datasets:
            raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(self.datasets.keys())}")
        
        # check 特徵的型態
        feature_type = self.feature_type_map.get(name, "unknown")  # fallback to "unknown" if not found
        dataset = self.datasets[name]()  # Load dataset
        print(f"Dataset: {dataset}:")
        # print(f"Number of graphs: {len(dataset)}")
        print(f"Number of features: {dataset.num_features}")
        print(f"Number of classes: {dataset.num_classes}")
        
        data = dataset[0]  # Retrieve first graph data
        num_ori_edges = data.num_edges
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


        # Initialize extra attributes (for consistency even without feature2node)
        num_edges = data.edge_index.shape[1]
        num_nodes = data.x.shape[0]

        # edge masks → match Feature2Node 格式
        data.node_node_mask = torch.ones((num_edges,), device=data.x.device, dtype=torch.int32)
        data.node_feat_mask = torch.zeros((num_edges,), device=data.x.device, dtype=torch.int32)

        # node masks → match Feature2Node 格式 (目前沒加 feature node → (num_nodes,))
        data.is_feature_node = torch.zeros((num_nodes,), device=data.x.device, dtype=torch.bool)
        data.is_original_node = torch.ones((num_nodes,), device=data.x.device, dtype=torch.bool)


        return data, dataset.num_features, dataset.num_classes, feature_type, num_ori_edges

if __name__ == "__main__":
    loader = GraphDatasetLoader()
    dataset_name = input(f"Enter dataset name {list(loader.datasets.keys())}: ")
    data = loader.load_dataset(dataset_name)
