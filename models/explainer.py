import os
import torch
from utils.device import DEVICE
import numpy as np
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.explain import ModelConfig
from torch_geometric.explain.config import ModelMode
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer, DummyExplainer
from .cf_explanation.utils.utils import get_neighbourhood, normalize_adj
from .cf_explanation.cf_explainer import CFExplainer


class SubgraphExplainer:
    """
    A flexible GNN explainer for node regression, supporting multiple algorithms (GNNExplainer, PGExplainer, DummyExplainer, CFExplainer).
    """

    def __init__(self, model_class, dataset, data, model_path, trial_name, split_id,
                 explainer_type="GNNExplainer", hop=2, epoch=100, lr=0.01,
                 run_mode="stage2", cf_beta=0.5):
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
        self.lr = lr
        self.dataset = dataset
        self.data = data
        self.model_path = model_path
        self.model_class = model_class
        self.trial_name = trial_name
        self.split_id = split_id  # 用於分組的 split_id
        self.run_mode = run_mode
        self.cf_beta = cf_beta
        self.explainer_type = explainer_type

        self.device = DEVICE
        self.model, self.model_config_dict = self._load_model()
        self.task_type = self._determine_task_type()
        self.explainer = self._explainer_setting()  # 設定 explainer

    
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
        # print("Full config loaded:", full_config)

        # 只取出模型需要的參數
        allowed_keys = ['in_channels', 'hidden_channels', 'hidden_channels1', 'hidden_channels2', 'out_channels']
        model_config = {k: v for k, v in full_config.items() if k in allowed_keys}

        model = self.model_class(**model_config).to(self.device)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        # 載入訓練好的權重
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()

        return model, model_config


    def _explainer_setting(self):
        """Sets up the explainer for node regression."""
        if self.explainer_type == "GNNExplainer":
            algorithm = GNNExplainer(epochs=self.epoch, num_hops=self.hop, lr=self.lr)
        elif self.explainer_type == "PGExplainer":
            algorithm = PGExplainer(epochs=self.epoch, num_hops=self.hop)
        elif self.explainer_type == "DummyExplainer":
            algorithm = DummyExplainer()
        elif self.explainer_type == "CFExplainer":
            return None  # 不走 PyG explainer API
        else:
            raise ValueError(f"Unsupported explainer type: {self.explainer_type}")

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

        if self.explainer_type == "CFExplainer":
            cf_info = self._run_cf_explainer(node_idx)
            if save:
                self._save_explanation(node_idx=node_idx, explanation=cf_info, explainer_type=self.explainer_type)
            return cf_info
        
        else: # 其餘 PyG 支援的 Explainer
            explanation = self.explainer(data.x, data.edge_index, index=node_idx)
            y_value = data.y[node_idx].item()  # 取得節點的回歸目標數值
            if save:
                self._save_explanation(node_idx=node_idx, explanation=explanation, explainer_type=self.explainer.algorithm.__class__.__name__)#, y_value=y_value)
            
            torch.cuda.empty_cache()
            return explanation.node_mask, explanation.edge_mask


    def _run_cf_explainer(self, node_idx, neighbor_threshold=10000):
        self.model = self.model.to(self.device)
        edge_index = self.data.edge_index
        features = self.data.x
        labels = self.data.y

        # 若有 feature to node，labels 需要補滿
        total_num_nodes = features.shape[0]  # 包含原始節點 + feature 節點（若轉成node）
        if labels.shape[0] < total_num_nodes:
            # 補成全圖大小，未標記部分設為 -1（或其他特別標記）
            all_labels = torch.full((total_num_nodes,), -1, dtype=labels.dtype, device=labels.device)
            all_labels[:labels.shape[0]] = labels  # 假設labels前面是原始節點
            labels = all_labels

        sub_edge_index, sub_feat, sub_labels, node_dict = get_neighbourhood(int(node_idx), edge_index, self.hop, features, labels)
        num_neighbors = sub_feat.shape[0]
        # 鄰居超過門檻，自動降為 1-hop
        if num_neighbors > neighbor_threshold and self.hop > 1:
            print(f"[Adjust] Too many neighbors ({num_neighbors}), retry with 1-hop.")
            sub_edge_index, sub_feat, sub_labels, node_dict = get_neighbourhood(int(node_idx), edge_index, 1, features, labels)
            num_neighbors = sub_feat.shape[0]
            print(f"Node {node_idx} with 1-hop has {num_neighbors} neighbors.")
   
        sub_edge_index = sub_edge_index.to(self.device)
        sub_feat = sub_feat.to(self.device)
        sub_labels = sub_labels.to(self.device)

        new_idx = node_dict[int(node_idx)]
        # print("node_dict:", node_dict)

        # 檢查是否為孤立節點（無邊情況）
        if sub_edge_index.size(1) == 0:
            print(f"Skip node {node_idx}: isolated node (no edges in subgraph).")
            return None
        
        # 轉 dense adjacency，並檢查 adj 是否異常
        sub_adj = to_dense_adj(sub_edge_index, max_num_nodes=sub_feat.size(0)).squeeze()

        # 這裡要注意 sub_adj 可能 squeeze 後變成 0-dim (完全沒有邊)
        if sub_adj.numel() == 0 or sub_adj.dim() < 2 or sub_adj.shape[0] == 0:
            print(f"Skip node {node_idx}: sub_adj is empty or invalid.")
            return None

        sub_adj = sub_adj.to(self.device)
        sub_edge_weight = torch.ones(sub_edge_index.size(1), device=self.device)
        self.model = self.model.to(self.device)

        y_pred_orig = self.model(sub_feat, sub_edge_index, sub_edge_weight)
        y_pred_orig = torch.argmax(y_pred_orig, dim=1) 

        self.cf_explainer = CFExplainer(model=self.model, sub_adj=sub_adj, sub_feat=sub_feat, 
                                sub_labels=sub_labels, y_pred_orig=y_pred_orig[new_idx], 
                                beta=self.cf_beta, device=self.device)

        cf_explanation = self.cf_explainer.explain(cf_optimizer="SGD", node_idx=node_idx, 
                                  new_idx=new_idx, lr=self.lr, 
                                  n_momentum=0.0, num_epochs=self.epoch, node_dict=node_dict)
        
        print("cf_explanation:", cf_explanation)

        torch.cuda.empty_cache()
        return cf_explanation


    def _save_explanation(self, node_idx, explanation, explainer_type):
        """
        Saves node ID, node mask, and edge mask into a structured folder hierarchy.
        """

        # Define save directory based on run mode, dataset, and explainer type
        save_exp_dir = os.path.join("saved", self.run_mode, f"split_{self.split_id}", explainer_type, self.dataset,  f"{self.trial_name}_{self.model_class.__name__}")
        os.makedirs(save_exp_dir, exist_ok=True)

        # Define file path for saving explanations
        file_path = os.path.join(save_exp_dir, f"node_{node_idx}.npz")
        fig_path = os.path.join(save_exp_dir, f"node_{node_idx}.png")

        if explainer_type == "CFExplainer":
            # Save counterfactual explanation
            if explanation is not None:
                np.savez_compressed(file_path, **explanation)
                print(f"Saved {explainer_type} explanation for node {node_idx} at {file_path}")
            else:
                print("No Counterfactual explanation is generated")
                
            if hasattr(self, "cf_explainer") and self.cf_explainer is not None:
                self.cf_explainer.plot_loss(fig_path)
        
        
        else:
            # Prepare explanation data
            explanation_data = {
                "model_name": self.model_class.__name__,
                "explainer_type": explainer_type,
                "task_type": self.task_type, 
                "node_idx": node_idx,
                # "y_value": y_value, # 好像用不到
            }

            # Convert masks to NumPy and reduce precision
            node_mask = explanation.node_mask.cpu().numpy().astype(np.float16) if explanation.node_mask is not None else None
            edge_mask = explanation.edge_mask.cpu().numpy().astype(np.float16) if explanation.edge_mask is not None else None

            # Save to compressed .npz file
            np.savez_compressed(file_path, **explanation_data, node_mask=node_mask, edge_mask=edge_mask)
            print(f"Saved {explainer_type} explanation for node {node_idx} at {file_path}")

