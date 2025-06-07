import os
import csv
import torch
import torch.nn as nn
import numpy as np

def min_max_norm(arr):
    """Min-max normalize a NumPy array to [0, 1]."""
    arr = np.array(arr, dtype=float)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)

def l1_normalize_rowwise(tensor):
    """L1 normalize each row of a tensor."""
    tensor_np = tensor.cpu().numpy()
    row_norm = np.sum(np.abs(tensor_np), axis=1, keepdims=True)
    row_norm[row_norm < 1e-12] = 1.0  # avoid div by 0
    tensor_np = tensor_np / row_norm
    return torch.tensor(tensor_np, device=tensor.device)

class StructureFeatureBuilder:
    def __init__(self, data, device, dataset_name: str,
                 feature_to_node: bool, only_feature_node: bool, only_structure: bool,
                 mode: str = "random+imp", emb_dim: int = 32, normalize_type: str = "row_l1",
                 save_dir: str = "saved/node_imp"):
        self.data = data
        self.dataset_name = dataset_name
        self.feature_to_node = feature_to_node
        self.only_feature_node = only_feature_node
        self.only_structure = only_structure
        self.mode = mode
        self.emb_dim = emb_dim
        self.normalize_type = normalize_type
        self.save_dir = save_dir
        self.device = device

        self.num_nodes = data.num_nodes

        # Node importance CSV path
        if self.feature_to_node:
            if self.only_feature_node:
                suffix = f"{self.dataset_name}_fn.csv"
            else:
                suffix = f"{self.dataset_name}_fn_nn.csv"
        
        else: # only_structure
            suffix = f"{self.dataset_name}_ori.csv"

        self.imp_csv_path = os.path.join(self.save_dir, self.dataset_name, suffix)

        # Embedding layer if using random+imp
        if self.mode == "random+imp":
            self.embedding = nn.Embedding(self.num_nodes, self.emb_dim) # 已經做了 feature_to_node，num_nodes 會包含 feature
            nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0) # 初始化

        # This will store the final feature dim after build()
        self.feature_dim = None

    def load_node_importance(self):
        node_imp = {}
        with open(self.imp_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = int(row['node'])
                degree = float(row['degree_norm'])
                pagerank = float(row['pagerank_norm'])
                betweenness = float(row['betweenness_norm'])
                closeness = float(row['closeness_norm'])
                node_imp[node_id] = [degree, pagerank, betweenness, closeness]

        imp_np = np.zeros((self.num_nodes, 4), dtype=float)
        for node_id, imp_vec in node_imp.items():
            imp_np[node_id] = np.array(imp_vec, dtype=float)

        # Per-column min-max normalize
        for j in range(4):
            imp_np[:, j] = min_max_norm(imp_np[:, j])

        imp_feat = torch.tensor(imp_np, device=self.device)
        return imp_feat


    def build(self):
        imp_feat = self.load_node_importance()

        # Prepare edge_type_feat if needed
        # 使用 feature to node，且 node-node 邊, node-feature 邊要一起看的時候需要多加兩維
        add_edge_type_feat = self.feature_to_node and not self.only_feature_node

        if add_edge_type_feat: # 只要用在需要存 (節點邊) (特徵邊) 的情況
            print("[StructureFeatureBuilder] Preparing edge_type_feat (2 dim)")
            edge_type_feat = torch.zeros((self.num_nodes, 2), device=self.device)
            if hasattr(self.data, 'is_feature_node'):
                for i in range(self.num_nodes):
                    if self.data.is_feature_node[i]:
                        edge_type_feat[i] = torch.tensor([0.0, 1.0], device=self.device) # node-feature 邊
                    else:
                        edge_type_feat[i] = torch.tensor([1.0, 0.0], device=self.device) # node-node 邊
        else:
            edge_type_feat = None  # 不需要用到

        # Build feature
        if self.mode == "one":
            print("[StructureFeatureBuilder] Mode: one → concat [1, imp]")
            one_feat = torch.ones((self.num_nodes, 1), device=self.device)
            base_feat = torch.cat([one_feat, imp_feat], dim=1)

        elif self.mode == "random+imp":
            node_ids = torch.arange(self.num_nodes, device=self.device)
            rand_embed = self.embedding(node_ids)
            print(f"[StructureFeatureBuilder] Mode: random+imp → rand_embed shape: {rand_embed.shape}")
            base_feat = torch.cat([rand_embed, imp_feat], dim=1)

        else:
            raise ValueError(f"Unknown mode {self.mode}, must be 'one' or 'random+imp'.")

        # Decide final concat
        if self.only_structure:
            print("[StructureFeatureBuilder] only_structure=True → no edge_type_feat → use base_feat")
            full_feat = base_feat

        elif self.feature_to_node and self.only_feature_node:
            print("[StructureFeatureBuilder] feature_to_node=True, only_feature_node=True → no edge_type_feat → use base_feat")
            full_feat = base_feat

        elif add_edge_type_feat: # feature_to_node and not self.only_feature_node:
            print("[StructureFeatureBuilder] feature_to_node=True, only_feature_node=False → concat edge_type_feat")
            full_feat = torch.cat([base_feat, edge_type_feat], dim=1)

        else:
            print("[StructureFeatureBuilder] No special condition → use base_feat")
            full_feat = base_feat

        # Row-wise L1 normalize if requested
        if self.normalize_type == "row_l1":
            full_feat = l1_normalize_rowwise(full_feat)
            print("[StructureFeatureBuilder] Applied row-wise L1 normalize.")

        # Store feature_dim
        self.feature_dim = full_feat.shape[1]
        print(f"[StructureFeatureBuilder] Built features shape: {full_feat.shape} → feature_dim={self.feature_dim}")

        return full_feat
    
    def get_edge_data(self):
        return extract_edges(self.data, self.feature_to_node, self.only_feature_node)
    

def extract_edges(data, feature_to_node: bool, only_feature_node: bool):
    """
    Extract correct edge_index and edge_weight from data, based on feature_to_node and only_feature_node flags.
    """
    # 處理 edge mask
    if feature_to_node:
        assert hasattr(data, "node_node_mask"), "Missing node_node_mask"
        assert hasattr(data, "node_feat_mask"), "Missing node_feat_mask"

        if only_feature_node:
            mask = data.node_feat_mask
            print("[EdgeExtractor] only_feature_node=True → using feature edges only")
        else:
            mask = data.node_node_mask | data.node_feat_mask
            print("[EdgeExtractor] using both node-node and feature edges")
        
        edge_index = data.edge_index[:, mask]
        edge_weight = data.edge_weight[mask] if hasattr(data, "edge_weight") else None
    
    else: # 原 graph
        print("[EdgeExtractor] using original edges")
        edge_index = data.edge_index
        edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None

    # 回傳
    return edge_index, edge_weight

