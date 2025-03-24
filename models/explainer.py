import os
import torch
import numpy as np
from torch_geometric.explain import ModelConfig
from torch_geometric.explain.config import ModelMode
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer, DummyExplainer

class SubgraphExplainer:
    """
    A flexible GNN explainer for node regression, supporting multiple algorithms (GNNExplainer, PGExplainer).
    """

    def __init__(self, model_class, dataset, data, model_path, trial_name, 
                 explainer_type="GNNExplainer", hop=2, epoch=100, 
                 run_mode="stage2", remove_feature=None, device=None, 
                 choose_nodes="random"):
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
        :param device: The device to use ('cpu' or 'cuda').
        :param task_type: Type of task ("regression" or "classification").
        """
        self.epoch = epoch
        self.hop = hop
        self.dataset = dataset
        self.data = data
        self.model_path = model_path
        self.model_class = model_class
        self.trial_name = trial_name
        self.run_mode = run_mode
        self.remove_feature = remove_feature
        self.choose_nodes = choose_nodes

        self.device = device if device else ("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.task_type = self._determine_task_type()
        self.explainer = self._explainer_setting(explainer_type)  # 設定 explainer

    
    def _determine_task_type(self):
        """Determines whether the model is for regression or classification based on its name."""
        model_name = self.model_class.__name__.lower()
        if "regressor" in model_name:
            return ModelMode.regression
        elif "classifier" in model_name:
            # return ModelMode.binary_classification if len(torch.unique(self.data.y)) == 2 else ModelMode.multiclass_classification
            # 由於我的 model 匯出是 #class 維度，直接使用 multiclass_classification 就好，用 binary 還要轉換...
            return ModelMode.multiclass_classification
        else:
            raise ValueError("Model class name must contain 'Regressor' or 'Classifier' to determine task type.")
        
        
    def _load_model(self):
        """Loads the model config and weights."""
        # 假設 config 檔名與 model_path 同名，只是多了 `_config`
        config_path = self.model_path.replace(".pth", "_config.pth")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config not found: {config_path}")

        full_config = torch.load(config_path, map_location=self.device)
        print("Full config loaded:", full_config)

        # 只取出模型需要的參數
        allowed_keys = ['in_channels', 'hidden_channels', 'hidden_channels1', 'hidden_channels2', 'out_channels']
        model_config = {k: v for k, v in full_config.items() if k in allowed_keys}

        model = self.model_class(**model_config).to(self.device)

        # 載入訓練好的權重
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()

        return model


    def _explainer_setting(self, explainer_type):
        """Sets up the explainer for node regression."""
        if explainer_type == "GNNExplainer":
            algorithm = GNNExplainer(epochs=self.epoch, num_hops=self.hop)
        elif explainer_type == "PGExplainer":
            algorithm = PGExplainer(epochs=self.epoch, num_hops=self.hop)
        elif explainer_type == "DummyExplainer":
            algorithm = DummyExplainer()
        else:
            raise ValueError(f"Unsupported explainer type: {explainer_type}")

        print(f"Task Type: {self.task_type}") 
        
        return Explainer(model=self.model, algorithm=algorithm, explanation_type='model',
                         node_mask_type='attributes', edge_mask_type='object',
                         model_config=ModelConfig(mode=self.task_type, task_level='node', 
                                                  return_type='probs' if "classification" in self.task_type.name else 'raw'))


    def explain_node(self, node_idx, data, save=True):
        """Explains a node and saves its explanation result."""
        self.model.eval()
        data = data.to(self.device)  # 確保數據也移動到對應裝置
        data.y = data.y.long()
        explanation = self.explainer(data.x, data.edge_index, index=node_idx) # 這邊問題

        y_value = data.y[node_idx].item()  # 取得節點的回歸目標數值

        if save:
            self._save_explanation(node_idx, explanation, self.explainer.algorithm.__class__.__name__, y_value)

        return explanation.node_mask, explanation.edge_mask



    def _save_explanation(self, node_idx, explanation, explainer_type, y_value):
        """
        Saves node ID, node mask, and edge mask into a structured folder hierarchy.
        """

        # Define save directory based on run mode, dataset, and explainer type
        save_exp_dir = os.path.join("saved", self.run_mode, explainer_type, self.dataset, self.choose_nodes, f"{self.trial_name}_{self.model_class.__name__}")
        os.makedirs(save_exp_dir, exist_ok=True)

        # Define file path for saving explanations
        file_path = os.path.join(save_exp_dir, f"node_{node_idx}.npz")

        # Prepare explanation data
        explanation_data = {
            "model_name": self.model_class.__name__,
            "explainer_type": explainer_type,
            "task_type": self.task_type, 
            "node_idx": node_idx,
            "y_value": y_value,
            "removed_feature_index": self.remove_feature
        }

        # Convert masks to NumPy and reduce precision
        node_mask = explanation.node_mask.cpu().numpy().astype(np.float16) if explanation.node_mask is not None else None
        edge_mask = explanation.edge_mask.cpu().numpy().astype(np.float16) if explanation.edge_mask is not None else None

        # Save to compressed .npz file
        np.savez_compressed(file_path, **explanation_data, node_mask=node_mask, edge_mask=edge_mask)

        print(f"Saved explanation for node {node_idx} at {file_path}")

