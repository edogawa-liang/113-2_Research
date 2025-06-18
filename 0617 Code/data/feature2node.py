# if variable is category: add feature to node
# if variable is continuous: add feature's node, feature value to edge weight

import torch
import numpy as np
from torch_geometric.data import Data


class FeatureNodeConverter:
    def __init__(self, feature_type, device):
        """
        :param is_categorical: True 表示特徵是類別型（0/1），False 表示連續型（浮點）
        """
        self.feature_type = feature_type
        self.device = device
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

        # 新增 feature node 的邊
        for node_id in range(num_nodes):
            for feat_id in range(num_features):
                feat_value = data.x[node_id, feat_id].item()
                feat_node_id = feature_node_offset + feat_id

                # if self.feature_type == "categorical":
                #     if feat_value == 1:  # 有值的才有邊連過去
                #         edge_index.append([node_id, feat_node_id])  # 單向：原節點 → 特徵節點
                #         edge_weight.append(1.0)
                #         node_node_mask.append(0)
                #         node_feat_mask.append(1)

                # elif self.feature_type == "continuous":
                #     edge_index.append([node_id, feat_node_id])  # 單向：原節點 → 特徵節點
                #     edge_weight.append(feat_value)
                #     node_node_mask.append(0)
                #     node_feat_mask.append(1)
                
                if self.feature_type == "categorical":
                    if feat_value == 1: # 有值的才有邊連過去
                        edge_index.append([node_id, feature_node_offset + feat_id]) # 原節點-> 特徵節點
                        edge_index.append([feature_node_offset + feat_id, node_id])  # 特徵節點-> 原節點
                        edge_weight.extend([1.0, 1.0])
                        node_node_mask.extend([0, 0])
                        node_feat_mask.extend([1, 1])

                elif self.feature_type == "continuous": # 每個 feature 會連到每個 node
                    edge_index.append([node_id, feat_node_id]) # 原節點-> 特徵節點
                    edge_index.append([feat_node_id, node_id]) # 特徵節點-> 原節點
                    edge_weight.extend([feat_value, feat_value])
                    node_node_mask.extend([0, 0])
                    node_feat_mask.extend([1, 1])


        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()

        # 處理 edge_weight → abs + normalize + clip
        edge_weight_np = np.array(edge_weight, dtype=float)
        edge_weight_np = np.abs(edge_weight_np)
        vmin, vmax = edge_weight_np.min(), edge_weight_np.max()
        edge_weight_np = (edge_weight_np - vmin) / (vmax - vmin + 1e-6)
        edge_weight_np = np.clip(edge_weight_np, 1e-6, 1.0)
        print(f"[Feature-to-Node] Processed edge weight range: min={edge_weight_np.min()}, max={edge_weight_np.max()}")
        edge_weight = torch.tensor(edge_weight_np, dtype=torch.float, device=self.device)
        
        node_node_mask = torch.tensor(node_node_mask, dtype=torch.bool, device=self.device)
        node_feat_mask = torch.tensor(node_feat_mask, dtype=torch.bool, device=self.device)


        # 節點特徵設為 1（原始 + feature node）(暫時)
        x = torch.ones((num_nodes + num_features, 1), device=self.device)

        # 標記原始節點及特徵節點
        is_feature_node = torch.zeros(num_nodes + num_features, dtype=torch.bool, device=self.device)
        is_feature_node[num_nodes:] = True
        is_original_node = torch.ones(num_nodes + num_features, dtype=torch.bool, device=self.device)
        is_original_node[num_nodes:] = False
        
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
