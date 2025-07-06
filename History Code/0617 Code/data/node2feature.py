# 將 feature to node 的 graph 產生的解釋子圖，轉回原本有 node feature 的 graph

# 需要再寫可以考慮特徵的 explainer_selector, remaining_graph (給GNNExplainer, DummyExplainer)
# random_selector 也要加可以random 移除特徵

import torch
from torch_geometric.data import Data
from collections import defaultdict


class FeatureNodeReverter:
    def __init__(self, feature_type, device):
        self.feature_type = feature_type
        self.device = device

    def revert(self, converted_data: Data, original_data: Data) -> Data:
        """
        將經過 feature-to-node 轉換的圖還原成含特徵的原始圖（保留所有欄位，沒被連接的補 0）。
        """
        num_nodes = original_data.num_nodes
        num_features = original_data.x.size(1)
        device = self.device

        x_restored = torch.zeros((num_nodes, num_features), device=device)

        edge_index = converted_data.edge_index
        src_all = edge_index[0]
        dst_all = edge_index[1]

        # 兩種 node-feature 邊: src 是 node, dst 是 feat node；或相反
        mask_src_node = (src_all < num_nodes) & (dst_all >= num_nodes) # node-feature 邊
        mask_dst_node = (dst_all < num_nodes) & (src_all >= num_nodes) # feature-node 邊

        # 總共 node-feature 的邊
        node_feat_mask = mask_src_node | mask_dst_node # 特徵邊

        # 節點和特徵的對應 index
        src = src_all[node_feat_mask]
        dst = dst_all[node_feat_mask]

        node_ids = torch.where(src < num_nodes, src, dst)
        feat_ids = torch.where(src >= num_nodes, src, dst) - num_nodes

        if self.feature_type == "categorical":
            x_restored[node_ids, feat_ids] = 1.0
        elif self.feature_type == "continuous":
            x_restored[node_ids, feat_ids] = original_data.x[node_ids, feat_ids]

        # 保留 node-node 邊
        nn_mask = (src_all < num_nodes) & (dst_all < num_nodes)
        edge_index_new = edge_index[:, nn_mask]
        edge_weight_new = converted_data.edge_weight[nn_mask] if hasattr(converted_data, "edge_weight") else None

        reverted_data = Data(x=x_restored, edge_index=edge_index_new, edge_weight=edge_weight_new, y=original_data.y)

        for attr in ["train_mask", "val_mask", "test_mask", "unknown_mask"]:
            if hasattr(original_data, attr):
                setattr(reverted_data, attr, getattr(original_data, attr))


        # 印出 zero 數量比較
        num_zero_original = (original_data.x == 0).sum().item()
        num_zero_restored = (x_restored == 0).sum().item()
        print(f"[FeatureNodeReverter] Zero count - original_data.x: {num_zero_original}, x_restored: {num_zero_restored}")

        num_one_original = (original_data.x == 1).sum().item()
        num_one_restored = (x_restored == 1).sum().item()
        print(f"[FeatureNodeReverter] One count - original_data.x: {num_one_original}, x_restored: {num_one_restored}")

        # original_data 0 的數量應該要比 x_restored 少


        return reverted_data

