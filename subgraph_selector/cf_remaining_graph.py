import os
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data

# 還沒跑過驗證
class CFSubgraphRemover:
    """
    Select edges to remove based on counterfactual explanations.
    """

    def __init__(self, data, base_dir, dataset_name, device, selected_nodes, fraction, top_k_percent_feat, only_feature_node):
        self.data = data.to(device)
        self.base_path = os.path.join(base_dir, "CFExplainer", dataset_name)
        self.device = device
        self.selected_nodes = selected_nodes  # 新增，可外部指定
        self.fraction = fraction
        self.top_k_percent_feat = top_k_percent_feat
        self.cf_removed_edges = []  # 儲存要移除的邊 (原始 edge_index 格式)
        self.only_feature_node = only_feature_node  # 是否只處理 feature node


    def load_data(self, trial_name):
        """
        從每個模型中找出被選到的節點，並載入對應的 cf_explanation（edge_index）來收集要移除的邊。
        """

        exp_dir = os.path.join(self.base_path, f"{trial_name}_GCN2Classifier")
        all_edges = []
        all_importance = []
        explained_nodes = []

        for node_id in self.selected_nodes:
            file_path = os.path.join(exp_dir, f"node_{node_id}.npz")
            
            if not os.path.exists(file_path):
                print(f"[Skip] Node {node_id} has no explanation.")
                continue

            data = np.load(file_path, allow_pickle=True)
            cf_explanation = data["cf_explanation"]
            edge_importance = data["edge_importance"]
            # 後續會一次移除雙向邊，不需要自補雙向邊
            
            all_edges.append(torch.tensor(cf_explanation, device=self.device))
            all_importance.append(torch.tensor(edge_importance, device=self.device))
            explained_nodes.append(node_id)

        # 有生成解釋的數量
        self.nodes_with_cf = len(explained_nodes)
        
        if all_edges:
            edges = torch.cat(all_edges, dim=1)
            importance = torch.cat(all_importance)

            edges_t = edges.t()
            unique_edges, unique_indices = torch.unique(edges_t, dim=0, return_inverse=True)
            edges = unique_edges.t()

            # 對應 importance：將 importance 加總
            importance_per_edge = torch.zeros(unique_edges.size(0), device=self.device)
            for i, idx in enumerate(unique_indices):
                importance_per_edge[idx] += importance[i]

            self.cf_removed_edges = edges
            self.edge_importance = importance_per_edge # 這裏的 edge 和 importance 是對應的

            print(f"[Summary] Loaded {edges.size(1)} unique CF edges from {len(explained_nodes)}/{len(self.selected_nodes)} nodes.")

        else:
            self.cf_removed_edges = torch.empty((2, 0), dtype=torch.long, device=self.device)
            self.edge_importance = torch.empty(0, device=self.device)
            print("No CF removed edges found.")


    def get_remaining_graph(self):
        num_total_edges = edge_index.size(1)
        node_node_mask = self.data.node_node_mask.bool()
        node_feat_mask = self.data.node_feat_mask.bool()
        num_node_edges = node_node_mask.sum().item()
        num_feat_edges = node_feat_mask.sum().item()


        # 先建 edge_map
        edge_index = self.data.edge_index
        edge_map = { (edge_index[0, i].item(), edge_index[1, i].item()): i for i in range(edge_index.size(1)) }

        cf_node_edges = []
        cf_feat_edges = []
        cf_importance_node = []
        cf_importance_feat = []

        for i, (u, v) in enumerate(self.cf_removed_edges.t()):
            
            idx = edge_map.get((u.item(), v.item()), None)
            if idx is None:
                continue  # 邊不存在，跳過

            if node_node_mask[idx]:
                cf_node_edges.append(idx)
                cf_importance_node.append(self.edge_importance[i].item())
            elif node_feat_mask[idx]:
                cf_feat_edges.append(idx)
                cf_importance_feat.append(self.edge_importance[i].item())


        # 原始圖裡第幾條邊
        cf_node_edges = torch.tensor(cf_node_edges, dtype=torch.long, device=self.device)
        cf_feat_edges = torch.tensor(cf_feat_edges, dtype=torch.long, device=self.device)
        cf_importance_node = np.array(cf_importance_node)
        cf_importance_feat = np.array(cf_importance_feat)

        k_node = int(self.fraction * num_node_edges)
        num_pairs = num_feat_edges // 2
        k_feat_pair = int(self.top_k_percent_feat * num_pairs)

        if k_node > len(cf_node_edges):
            print("[Warning] Not enough node edges, removing all available node edges.")
            k_node = len(cf_node_edges)

        if k_feat_pair * 2 > len(cf_feat_edges):
            print("[Warning] Not enough feature edges, removing all available feature edges.")
            k_feat_pair = len(cf_feat_edges) // 2

        idx_remove_node = cf_node_edges[cf_importance_node.argsort()[-k_node:]] if k_node > 0 else torch.empty(0, dtype=torch.long, device=self.device)

        pair_scores = []
        feat_edge_indices = cf_feat_edges # 該條邊在 self.data.edge_index 裡的位置
        feat_importance = cf_importance_feat



        # 動態計算 offset
        if self.only_feature_node:
            feat_edge_start = 0
        else:
            feat_edge_start = self.data.node_node_mask.sum().item()

        # pair_scores 計算改寫：
        for p_idx in range(num_pairs):
            idx1 = feat_edge_start + p_idx * 2
            idx2 = feat_edge_start + p_idx * 2 + 1

            mask1 = (feat_edge_indices == idx1)
            mask2 = (feat_edge_indices == idx2)

            score1 = feat_importance[mask1.nonzero(as_tuple=True)[0]].max().item() if mask1.any() else 0
            score2 = feat_importance[mask2.nonzero(as_tuple=True)[0]].max().item() if mask2.any() else 0

            pair_scores.append((score1 + score2) / 2)

        pair_scores = np.array(pair_scores)
        top_pair_idx = np.argsort(pair_scores)[-k_feat_pair:] if k_feat_pair > 0 else []

        idx_remove_feat = torch.cat([
            torch.tensor([feat_edge_start + p_idx * 2 for p_idx in top_pair_idx], device=self.device),
            torch.tensor([feat_edge_start + p_idx * 2 + 1 for p_idx in top_pair_idx], device=self.device)
        ]) if k_feat_pair > 0 else torch.empty(0, dtype=torch.long, device=self.device)

        idx_keep = torch.ones(num_total_edges, dtype=torch.bool, device=self.device)
        idx_keep[idx_remove_node] = False
        idx_keep[idx_remove_feat] = False
        new_edge_index = edge_index[:, idx_keep]
        print(f"Removed {len(idx_remove_node)} node edges and {len(idx_remove_feat)} feature edges.")


        # 統計資訊
        stats = {
            "selected_nodes": len(self.selected_nodes),
            "nodes_with_cf": self.nodes_with_cf,
            "nodes_without_cf": len(self.selected_nodes) - self.nodes_with_cf,
            "cf_node_edge_percent": (len(cf_node_edges) / num_node_edges * 100) if num_node_edges > 0 else 0,
            "cf_feat_edge_percent": (len(cf_feat_edges) / num_feat_edges * 100) if num_feat_edges > 0 else 0,
            "removed_node_edge_percent": (len(idx_remove_node) / num_node_edges * 100) if num_node_edges > 0 else 0,
            "removed_feat_edge_percent": (len(idx_remove_feat) / num_feat_edges * 100) if num_feat_edges > 0 else 0
        }

        print(f"[CF Summary] {stats}")

        return Data(x=self.data.x, edge_index=new_edge_index, y=self.data.y, 
                    train_mask=self.data.train_mask, val_mask=self.data.val_mask, test_mask=self.data.test_mask, unknown_mask=self.data.unknown_mask
                    ), stats
