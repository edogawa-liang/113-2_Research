# if variable is category: add feature to node
# if variable is continuous: add feature's node, feature value to edge weight

import torch
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
        feature_node_offset = num_nodes  # 特徵節點 ID 從這開始

        edge_index = []
        edge_weight = []

        for node_id in range(num_nodes):
            for feat_id in range(num_features):
                feat_value = data.x[node_id, feat_id].item()
                if self.feature_type == "categorical":
                    if feat_value == 1:  # 只連有值的特徵
                        edge_index.append([node_id, feature_node_offset + feat_id])
                        edge_index.append([feature_node_offset + feat_id, node_id])
                        edge_weight.extend([1.0, 1.0])

                elif self.feature_type == "continuous":
                    # 所有節點都連全部 feature node，邊權是原始特徵值
                    edge_index.append([node_id, feature_node_offset + feat_id])
                    edge_index.append([feature_node_offset + feat_id, node_id])

                    # 想一下怎麼改邊的權重?
                    # 暫時先不改，特徵保持負，傳遞訊息時讓自己越來越不像鄰居
                    edge_weight.extend([feat_value, feat_value])

        # 加上原始的 edge（補上 edge_weight = 1.0）
        if not hasattr(data, 'edge_weight') or data.edge_weight is None:
            original_edge_weight = torch.ones(data.edge_index.size(1), device=self.device)
        else:
            original_edge_weight = data.edge_weight

        # 合併 edge_index 與 edge_weight
        new_edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
        new_edge_weight = torch.tensor(edge_weight, dtype=torch.float, device=self.device)

        edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
        edge_weight = torch.cat([original_edge_weight, new_edge_weight], dim=0)

        # 節點特徵都設為 1（包含原始 + 特徵節點）
        x = torch.ones((num_nodes + num_features, 1), device=self.device)

        # 建立新圖
        new_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=data.y)
        # print(f"Original graph: {data}")
        print(f"Converted graph: {new_data}")

        # 複製並擴展 mask 屬性
        for attr in ["train_mask", "val_mask", "test_mask"]:
            if hasattr(data, attr):
                old_mask = getattr(data, attr)
                if old_mask is not None and old_mask.shape[0] == num_nodes:
                    pad = torch.zeros(num_features, dtype=torch.bool, device=old_mask.device)
                    new_mask = torch.cat([old_mask, pad], dim=0)
                    setattr(data, attr, new_mask)

        return new_data
