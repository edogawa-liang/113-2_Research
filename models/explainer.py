import os
import torch
import pickle
import importlib
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer


class SubgraphExplainer:
    """
    A flexible GNN explainer supporting multiple algorithms (GNNExplainer, PGExplainer).
    """

    def __init__(self, model_module, dataset, model_path, use_data="Facebook", explainer_type="GNNExplainer", hop=2, epoch=100, save_dir="saved/explain_subgraphs"):
        """
        Initializes the explainer.

        :param model_module: The module where the GNN model is stored (e.g., basic_GCN).
        :param dataset: The dataset.
        :param model_path: Path to the trained GNN model checkpoint (.pt file).
        :param use_data: Dataset name ("Facebook" or others).
        :param explainer_type: Type of explainer ("GNNExplainer" or "PGExplainer").
        :param hop: Number of hops for neighborhood expansion.
        :param epoch: Number of training epochs.
        :param save_dir: Directory to save explanations.
        """
        self.use_data = use_data
        self.epoch = epoch
        self.hop = hop
        self.dataset = dataset
        self.save_dir = save_dir
        self.model_module = model_module
        self.model_path = model_path
        self.model = self._load_model()  # Load model & weights
        self.explainer = self._explainer_setting(explainer_type)

        os.makedirs(self.save_dir, exist_ok=True)


    def _load_model(self):
        """Loads the GNN model from basic_GCN.py dynamically and loads weights."""

        importlib.reload(self.model_module)  # Reload module to ensure latest changes
        model_class_name = "GCN3" if self.hop == 3 else "GCN2"
        model_class = getattr(self.model_module, model_class_name, None)

        if model_class is None:
            raise ValueError(f"{model_class_name} model not found in the provided module.")

        # Initialize model
        model = model_class(in_channels=self.dataset.num_features, out_channels=self.dataset.num_classes)

        # Load trained weights
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            print(f"Loaded trained model weights from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint {self.model_path} not found!")

        return model


    def _explainer_setting(self, explainer_type):
        """Sets up the explainer."""
        if explainer_type == "GNNExplainer":
            algorithm = GNNExplainer(epochs=self.epoch, num_hops=self.hop)
        elif explainer_type == "PGExplainer":
            algorithm = PGExplainer(epochs=self.epoch)
        else:
            raise ValueError(f"Unsupported explainer type: {explainer_type}")

        return Explainer(
            model=self.model,
            algorithm=algorithm,
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
            ),
        )

    def explain_node(self, node_idx, data, save=True):
        """Explains a node and saves its explanation result."""
        self.model.eval()
        explanation = self.explainer(data.x, data.edge_index, index=node_idx)

        y_label = data.y[node_idx].item()  # Get node label

        if save:
            self._save_explanation(node_idx, explanation, self.use_data, self.explainer.algorithm.__class__.__name__, y_label)

        return explanation.node_mask, explanation.edge_mask


    def _save_explanation(self, node_idx, explanation, dataset_name, explainer_type, y):
        """
        Saves node ID, node mask, and edge mask into a structured folder hierarchy.
        """
        save_path = os.path.join(self.save_dir, dataset_name, explainer_type, str(y))
        os.makedirs(save_path, exist_ok=True)

        file_name = f"node{node_idx}.pkl"
        file_path = os.path.join(save_path, file_name)

        explanation_data = {
            "dataset": dataset_name,
            "explainer_type": explainer_type,
            "node_idx": node_idx,
            "y": y,
            "node_mask": explanation.node_mask.cpu().tolist() if explanation.node_mask is not None else None,
            "edge_mask": explanation.edge_mask.cpu().tolist() if explanation.edge_mask is not None else None,
        }

        with open(file_path, "wb") as f:
            pickle.dump(explanation_data, f)

        print(f"Saved explanation for node {node_idx} at {file_path}")





# Example Usage
if __name__ == "__main__":
    import basic_GCN
    from ..data.dataset_loader import GraphDatasetLoader
    from ..data.data_modifier import GraphModifier

    loader = GraphDatasetLoader()
    dataset_name = input(f"Enter dataset name {list(loader.datasets.keys())}: ")
    data = loader.load_dataset(dataset_name)
    trained_model_path = "saved/models/gnn_checkpoint.pt" # 改

    # 不用Pca 但還是要調 Dataset 根據training GNN 資料夾存法


    # Initialize explainer with trained weights
    explainer = SubgraphExplainer(
        model_module=basic_GCN,
        dataset=data,
        model_path=trained_model_path,
        use_data="Facebook",
        explainer_type="GNNExplainer",
        hop=3,
        epoch=100
    )

    # Pick nodes to explain
    node_indices = [10, 15, 25]

    # Explain each node and save results
    for node_idx in node_indices:
        print(f"\nExplaining node {node_idx}...")
        explainer.explain_node(node_idx, data, save=True)