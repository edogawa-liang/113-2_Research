import os
import torch
import pickle
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer

class SubgraphExplainer:
    """
    A flexible GNN explainer for node regression, supporting multiple algorithms (GNNExplainer, PGExplainer).
    """

    def __init__(self, model_class, dataset, data, model_path, trial_name, 
                 explainer_type="GNNExplainer", hop=2, epoch=100, 
                 run_mode="stage2_expsubg", config=None, remove_feature=None, device=None):
        """
        Initializes the explainer.

        :param model_class: The GNN model class (e.g., GCN2Regressor or GCN3Regressor).
        :param data: The data.
        :param model_path: Path to the trained GNN model checkpoint (.pth file).
        :param explainer_type: Type of explainer ("GNNExplainer" or "PGExplainer").
        :param hop: Number of hops for neighborhood expansion.
        :param epoch: Number of training epochs.
        :param run_mode: Run mode (base folder for saving explanations).
        :param trial_name: Name of the trial for saving explanations.
        :param config: Additional configuration for the model.
        :param device: The device to use ('cpu' or 'cuda').
        """
        self.epoch = epoch
        self.hop = hop
        self.dataset = dataset
        self.data = data
        self.model_path = model_path
        self.model_class = model_class
        self.trial_name = trial_name
        self.run_mode = run_mode
        self.config = config if config else {}
        self.remove_feature = remove_feature

        # 設定裝置
        self.device = device if device else ("cuda:1" if torch.cuda.is_available() else "cpu")

        # 載入模型
        self.model = self._load_model()

        # 設定 explainer
        self.explainer = self._explainer_setting(explainer_type)


    def _load_model(self):
        """Loads the GNN regression model and weights."""
        num_features = self.data.x.shape[1]
        model = self.model_class(in_channels=num_features, **self.config).to(self.device)

        # Load trained weights
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"Loaded trained model weights from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint {self.model_path} not found!")

        return model


    def _explainer_setting(self, explainer_type):
        """Sets up the explainer for node regression."""
        if explainer_type == "GNNExplainer":
            algorithm = GNNExplainer(epochs=self.epoch, num_hops=self.hop)
        elif explainer_type == "PGExplainer":
            algorithm = PGExplainer(epochs=self.epoch, num_hops=self.hop)
        else:
            raise ValueError(f"Unsupported explainer type: {explainer_type}")

        return Explainer(
            model=self.model,
            algorithm=algorithm,
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='node',
                return_type='raw',
            ),
        )


    def explain_node(self, node_idx, data, save=True):
        """Explains a node and saves its explanation result."""
        self.model.eval()
        data = data.to(self.device)  # 確保數據也移動到對應裝置

        explanation = self.explainer(data.x, data.edge_index, index=node_idx)

        y_value = data.y[node_idx].item()  # 取得節點的回歸目標數值

        if save:
            self._save_explanation(node_idx, explanation, self.explainer.algorithm.__class__.__name__, y_value)

        return explanation.node_mask, explanation.edge_mask


    def _save_explanation(self, node_idx, explanation, explainer_type, y_value):
        """
        Saves node ID, node mask, and edge mask into a structured folder hierarchy.
        """

        # Define save directory based on run mode, dataset, and explainer type
        save_exp_dir = os.path.join("saved", self.run_mode, explainer_type,self.dataset)
        os.makedirs(save_exp_dir, exist_ok=True)

        # Define file path for saving explanations
        file_path = os.path.join(save_exp_dir, f"{self.trial_name}_{self.model_class.__name__}_node{node_idx}.pkl")

        # Prepare explanation data
        explanation_data = {
            "model_name": self.model_class.__name__,
            "explainer_type": explainer_type,
            "node_idx": node_idx,
            "y_value": y_value,
            "removed_feature_index": self.remove_feature,
            "node_mask": explanation.node_mask.cpu().numpy() if explanation.node_mask is not None else None,
            "edge_mask": explanation.edge_mask.cpu().numpy() if explanation.edge_mask is not None else None,
            }
        # Save to file
        with open(file_path, "wb") as f:
            pickle.dump(explanation_data, f)

        print(f"Saved explanation for node {node_idx} at {file_path}")
