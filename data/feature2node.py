# if variable is category: add feature to node
# if variable is continuous: add feature's node, feature value to edge weight
# 可使用於單一類別型 或 連續型資料 (混合資料要另外設定)
# 讓模型自己學這個節點更看結構還是特徵

import torch
import numpy as np
from torch_geometric.data import Data

# 類別型和連續型都將 feature 轉為 edge_weight

class FeatureNodeConverter(torch.nn.Module):
    def __init__(self, feature_type, num_nodes, device):
        """
        :param is_categorical: True 表示特徵是類別型（0/1），False 表示連續型（浮點）
        """
        super().__init__()
        self.feature_type = feature_type
        self.device = device
        # self.feature_importance = feature_importance # >1 強化特徵邊，<1 削弱特徵邊
        self.node_feature_vs_structure = torch.nn.Parameter(torch.ones(num_nodes, device=device))

        print("dataset has", self.feature_type, "features")

    def convert(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        num_features = data.x.size(1)

        feature_node_offset = num_nodes # 特徵節點的排序是原節點後

        edge_index = []
        edge_weight = []
        node_node_mask = []
        node_feat_mask = []

        # 原始邊與邊權
        if not hasattr(data, 'edge_weight') or data.edge_weight is None:
            original_edge_weight = torch.ones(data.edge_index.size(1), device=self.device)
        else:
            original_edge_weight = data.edge_weight


        # 加入原始邊
        edge_index_orig = data.edge_index
        edge_index.extend(edge_index_orig.t().tolist())
        edge_weight.extend(original_edge_weight.tolist())
        node_node_mask.extend([1] * edge_index_orig.size(1)) # 原來的邊
        node_feat_mask.extend([0] * edge_index_orig.size(1))

        # 預取特徵矩陣
        feat_matrix = data.x.abs().clone().cpu().numpy()  # abs 處理
        feat_matrix = np.clip(feat_matrix, a_min=0.0, a_max=None)

        # 計算平均度數 (用於邊權計算)
        # avg_degree = data.edge_index.size(1) / num_nodes    

        # 計算每個節點 degree
        degree_count = torch.bincount(data.edge_index[0], minlength=num_nodes).cpu().numpy()

        # 希望每個節點公平看待結構和特徵
        if self.feature_type == "categorical":
            for node_id in range(num_nodes):
                node_degree = degree_count[node_id] + 1e-6  # 防止除零
                node_ones = np.sum(feat_matrix[node_id, :] == 1)
                if node_ones > 0:
                    # edge_w = (node_degree / node_ones) * self.feature_importance
                    # edge_w = (node_degree / node_ones) * self.node_feature_vs_structure[node_id]
                    edge_w = (node_degree / node_ones) * torch.clamp(self.node_feature_vs_structure[node_id], min=0.0)
                    edge_w = edge_w.item()
                else:
                    edge_w = 1.0  # 沒有特徵為 1，邊權不使用

                for feat_id in range(num_features):
                    if feat_matrix[node_id, feat_id] == 1:
                        feat_node_id = feature_node_offset + feat_id
                        edge_index.append([node_id, feat_node_id])
                        edge_index.append([feat_node_id, node_id])
                        edge_weight.extend([edge_w, edge_w])
                        node_node_mask.extend([0, 0])
                        node_feat_mask.extend([1, 1])

        elif self.feature_type == "continuous":

            # column-wise min-max normalize (加下界0.05防止過小邊權)
            col_min = feat_matrix.min(axis=0, keepdims=True)
            col_max = feat_matrix.max(axis=0, keepdims=True)
            norm_feat = (feat_matrix - col_min) / (col_max - col_min + 1e-6)
            norm_feat = np.clip(norm_feat, 0.05, 1.0)

            # row-wise L1 normalize
            row_sum = norm_feat.sum(axis=1, keepdims=True) + 1e-8
            norm_feat = norm_feat / row_sum

            # 乘上平均degree (暫時設定讓對於同一個節點，結構和特徵對其的影響差不多重要)
            # norm_feat *= avg_degree

            # 新增 feature node 的邊
            for node_id in range(num_nodes):
                node_degree = degree_count[node_id] + 1e-6

                for feat_id in range(num_features):
                    feat_value = norm_feat[node_id, feat_id]
                    if feat_value > 0:
                        # feat_value = feat_value * node_degree * self.feature_importance
                        # feat_value = feat_value * node_degree * self.node_feature_vs_structure[node_id]
                        feat_value = feat_value * node_degree * torch.clamp(self.node_feature_vs_structure[node_id], min=0.0)
                        feat_value = feat_value.item()

                        feat_node_id = feature_node_offset + feat_id
                        edge_index.append([node_id, feat_node_id])
                        edge_index.append([feat_node_id, node_id])
                        edge_weight.extend([feat_value, feat_value])
                        node_node_mask.extend([0, 0])
                        node_feat_mask.extend([1, 1])

        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float, device=self.device)
        node_node_mask = torch.tensor(node_node_mask, dtype=torch.bool, device=self.device)
        node_feat_mask = torch.tensor(node_feat_mask, dtype=torch.bool, device=self.device)

        # 標記原始節點及特徵節點
        is_feature_node = torch.zeros(num_nodes + num_features, dtype=torch.bool, device=self.device)
        is_feature_node[num_nodes:] = True
        is_original_node = torch.ones(num_nodes + num_features, dtype=torch.bool, device=self.device)
        is_original_node[num_nodes:] = False


        # 節點特徵設為 1（原始 + feature node）(暫時)
        x = torch.ones((num_nodes + num_features, 1), device=self.device)

        # 印出 edgeweight 統計資訊
        print("Edge weight statistics:")
        print(f"  New edge weight: {edge_weight.min().item()} - {edge_weight.max().item()} (mean: {edge_weight.mean().item()})")

        
        new_data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=data.y,
            node_node_mask=node_node_mask,
            node_feat_mask=node_feat_mask,
            is_feature_node=is_feature_node,
            is_original_node=is_original_node
        )

        return new_data

        # # 延伸 mask
        # for attr in ["train_mask", "val_mask", "test_mask", "unknown_mask"]:
        #     if hasattr(data, attr):
        #         old_mask = getattr(data, attr)
        #         if old_mask is not None and old_mask.shape[0] == num_nodes:
        #             pad = torch.zeros(num_features, dtype=torch.bool, device=old_mask.device)
        #             new_mask = torch.cat([old_mask, pad], dim=0)
        #             setattr(new_data, attr, new_mask)

        # print(f"Converted graph: {new_data}")
        # return new_data
