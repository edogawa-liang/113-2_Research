import os
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data

# 還沒+ feature_to_node 的情況

class CFSubgraphRemover:
    """
    Select edges to remove based on counterfactual explanations.
    """

    def __init__(self, data, base_dir, explainer_name, dataset_name, node_choose, device="cpu"):
        self.data = data.to(device)
        self.base_path = os.path.join(base_dir, explainer_name, dataset_name)
        self.sub_dirs = sorted([d for d in os.listdir(self.base_path) if d.endswith("GCN2Classifier")])
        self.device = device
        self.node_choose = node_choose
        self.cf_removed_edges = []  # 儲存要移除的邊 (原始 edge_index 格式)

    def load_data(self):
        """
        從每個模型中找出被選到的節點，並載入對應的 cf_explanation（edge_index）來收集要移除的邊。
        """
        self.node_count = 0

        for sub_dir in self.sub_dirs:
            model_dir = os.path.join(self.base_path, sub_dir)
            share_node_dir = os.path.join(model_dir, "share_nodes")
            csv_path = os.path.join(model_dir, "node_record.csv")

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"'{share_node_dir}' not found. Please run stage2_node_share first.")

            df = pd.read_csv(csv_path)
            if self.node_choose not in df.columns:
                raise ValueError(f"Column '{self.node_choose}' not found in {csv_path}. Please check if stage2_node_share has been correctly executed.")

            selected_nodes = df[df[self.node_choose] == 1]["Node"].tolist()
            if self.node_count == 0:
                self.node_count = len(selected_nodes)

            for node_id in selected_nodes:
                file_path = os.path.join(share_node_dir, f"node_{node_id}.npz")
                if not os.path.exists(file_path):
                    print(f"File {file_path} does not exist.")
                    continue

                data = np.load(file_path, allow_pickle=True)
                cf_explanation = data["cf_explanation"]
                edges = torch.tensor(cf_explanation, device=self.device)
                self.cf_removed_edges.append(edges)

        if self.cf_removed_edges:
            self.cf_removed_edges = torch.cat(self.cf_removed_edges, dim=1)  # [2, num_total_edges]

            # 去重複（轉成 [num_edges, 2]，使用 unique）
            edges_t = self.cf_removed_edges.t()  # [num_edges, 2]
            edges_t_unique = torch.unique(edges_t, dim=0)
            self.cf_removed_edges = edges_t_unique.t()  # 還原為 [2, num_edges]
            print(f"Found {self.cf_removed_edges.size(1)} CF removed edges.")   

        else:
            self.cf_removed_edges = torch.empty((2, 0), dtype=torch.long, device=self.device)
            print("No CF removed edges found.")

    def get_node_count(self):
        print(f"Number of nodes selected: {self.node_count}")
        return self.node_count

    def get_edge_count(self):
        print(f"Number of edges selected: {self.cf_removed_edges.size(1)}")
        return self.cf_removed_edges.size(1)

    def get_remaining_graph(self):
        """
        從原圖中移除 cf_removed_edges 中的邊，回傳新的 PyG Data。
        """
        edge_index = self.data.edge_index

        # 建立 set 方便比對
        remove_set = { (u.item(), v.item()) for u, v in self.cf_removed_edges.t() }
        # 注意 PyG 中 edge_index 是雙向的（若是 undirected graph）
        keep_mask = [
            i for i in range(edge_index.size(1))
            if (edge_index[0, i].item(), edge_index[1, i].item()) not in remove_set
        ]

        new_edge_index = edge_index[:, keep_mask]

        return Data(
            x=self.data.x,
            edge_index=new_edge_index,
            y=self.data.y,
            train_mask=self.data.train_mask,
            val_mask=self.data.val_mask,
            test_mask=self.data.test_mask
        )
