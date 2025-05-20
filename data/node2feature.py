# 將 feature to node 的 graph 產生的解釋子圖，轉回原本有 node feature 的 graph

# 需要再寫可以考慮特徵的 explainer_selector, remaining_graph (給GNNExplainer, DummyExplainer)
# random_selector 也要加可以random 移除特徵

import torch
from torch_geometric.data import Data

class FeatureNodeReverter:
    def __init__(self, feature_type, device):
        self.feature_type = feature_type
        self.device = device

    def revert(self, converted_data: Data, original_data: Data) -> Data:
        """
        將經過 feature-to-node 轉換的圖還原成含特徵的原始圖。
        
        :param converted_data: convert 後的圖（包含特徵節點）
        :param original_data: 原始圖，用來取得 node 數量與 feature 數量
        :return: 還原後的圖
        """
        num_nodes = original_data.num_nodes
        num_features = original_data.x.size(1)
        device = self.device

        x_restored = torch.zeros((num_nodes, num_features), device=device)

        for i in range(converted_data.edge_index.size(1)):
            src, dst = converted_data.edge_index[:, i]
            weight = converted_data.edge_weight[i]

            # 處理 ori node 連到 feature node 的狀況
            if src < num_nodes and dst >= num_nodes: 
                node_id = src.item()
                feat_id = dst.item() - num_nodes
            elif dst < num_nodes and src >= num_nodes: # 處理雙向邊的情況
                node_id = dst.item()
                feat_id = src.item() - num_nodes

            if self.feature_type == "categorical":
                x_restored[node_id, feat_id] = 1.0
            elif self.feature_type == "continuous":
                x_restored[node_id, feat_id] = weight

 
        # 如果特徵全為0, 則移除此特徵
        feature_sums = x_restored.sum(dim=0)
        zero_feature_cols = (feature_sums == 0).nonzero(as_tuple=True)[0].tolist()
        print(f"Number of removed feature columns (all-zero): {len(zero_feature_cols)}")
        x_restored = x_restored[:, feature_sums != 0]
 
        # 保留原始 node-node 邊與邊權重
        # < num_nodes 的就是原始邊
        mask = (converted_data.edge_index[0] < num_nodes) & (converted_data.edge_index[1] < num_nodes)
        edge_index = converted_data.edge_index[:, mask]
        edge_weight = converted_data.edge_weight[mask] if hasattr(converted_data, 'edge_weight') else None

        reverted_data = Data(x=x_restored, edge_index=edge_index, edge_weight=edge_weight, y=original_data.y)

        # 補上 mask
        for attr in ["train_mask", "val_mask", "test_mask"]:
            if hasattr(original_data, attr):
                setattr(reverted_data, attr, getattr(original_data, attr))


        return reverted_data, zero_feature_cols
