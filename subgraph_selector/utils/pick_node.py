# pick_node.py

import torch
import pandas as pd
import numpy as np
import os
import random
from torch_geometric.utils import k_hop_subgraph

class NodePicker:
    def __init__(self, data, dataset_name, node_choose, 
                 feature_to_node=False, only_feature_node=False, # load importance 需要
                 imp_dir="saved/node_imp", node_ratio="auto", edge_ratio=0.5):
        self.data = data
        self.dataset_name = dataset_name
        self.node_choose = node_choose
        self.feature_to_node = feature_to_node
        self.only_feature_node = only_feature_node
        self.imp_dir = imp_dir
        self.node_ratio = node_ratio
        self.edge_ratio = edge_ratio

        # 只看原始 node (只挑原始節點)
        is_original_node = data.is_original_node.cpu().numpy()
        self.original_node_indices = np.where(is_original_node)[0]

        # 只看原始 node-node edge (挑選節點數量時，只參考原始邊做計算)
        node_node_mask = data.node_node_mask.cpu().numpy()
        self.num_ori_edges = node_node_mask.sum()

        # 原始 node 數量 N
        self.num_ori_nodes = len(self.original_node_indices)
        print(f"Original nodes: {self.num_ori_nodes}, Original edges: {self.num_ori_edges}")

    def load_importance(self):
        if self.feature_to_node:
            if self.only_feature_node:
                suffix = f"{self.dataset_name}_fn.csv"
            else:
                suffix = f"{self.dataset_name}_fn_nn.csv"
        else:
            suffix = f"{self.dataset_name}_ori.csv"

        imp_path = os.path.join(self.imp_dir, self.dataset_name, suffix)
        print(f"Loading importance from {imp_path}")
        df = pd.read_csv(imp_path)

        # 應該有 columns: node, Degree, PageRank, Betweenness, Closeness
        return df

    # 只會從原始 node 中挑
    def pick_nodes(self):
        train_mask = self.data.train_mask.cpu().numpy()
        train_node_indices = np.where(train_mask)[0]

        # 只挑原始 node + train mask 進入 pool
        train_original_node_indices = np.intersect1d(train_node_indices, self.original_node_indices)

        if self.node_choose == "all_train":
            selected_nodes = train_original_node_indices.tolist()
            return selected_nodes

        # Load importance
        df = self.load_importance()
        df = df[df['node'].isin(train_original_node_indices)] # 在train內的所有importance

        # 決定要挑多少 nodes
        target_ratio = self._calculate_node_ratio(
            node_ratio=self.node_ratio,
            edge_ratio=self.edge_ratio
        )
        
        N = int(self.num_ori_nodes * target_ratio)
        print(f"Picking {N} nodes ({target_ratio * 100:.1f}%)")
        max_available = len(df['node'])
        N_final = min(N, max_available)
        if N > max_available:
            print(f"[Warning] Requested N={N}, but only {max_available} available nodes. Using all available nodes.")


        # Sort / Pick
        if self.node_choose == "random":
            selected_nodes = df['node'].sample(frac=1, random_state=42).tolist()[:N_final]

        elif self.node_choose == "stratified_by_degree":
            selected_nodes = self._select_stratified_by_degree(df['node'].tolist(), N_final)

        else:
            # Mapping node_choose to real column name
            node_choose_to_column = {
                "degree": "degree_norm",
                "pagerank": "pagerank_norm",
                "betweenness": "betweenness_norm",
                "closeness": "closeness_norm"
            }

            assert self.node_choose in node_choose_to_column, f"Unknown node_choose! Must be one of {list(node_choose_to_column.keys())}"
            column_name = node_choose_to_column[self.node_choose]

            assert column_name in df.columns, f"Column {column_name} not found in df.columns!"

            selected_nodes = df.sort_values(by=column_name, ascending=False)['node'].tolist()[:N_final]

        return selected_nodes




    def _calculate_node_ratio(self, node_ratio="auto", edge_ratio=0.5):
        """
        計算需要選擇的節點比例。
        會用原始的graph的平均度數來計算，而不會使用到 feature node。

        - node_ratio="auto"：根據 edge_ratio 估算需要多少節點
        - node_ratio=數值：直接使用傳入的節點比例
        - edge_ratio： 幫助計算 node_ratio
        """

        avg_degree = self.num_ori_edges / self.num_ori_nodes
        print(f"Average node degree (original graph): {avg_degree:.4f}")

        if node_ratio == "auto":
            target_edges = edge_ratio * self.num_ori_edges  # 目標邊數
            num_selected_nodes = target_edges / (avg_degree ** 2)      # 估算需要的節點數
            node_ratio_computed = num_selected_nodes / self.num_ori_nodes

            print(f"{num_selected_nodes:.2f} nodes required to ensure {edge_ratio * 100:.1f}% edges in the subgraph.")
            print(f"Computed node ratio: {node_ratio_computed:.4f}")

            return node_ratio_computed

        else:
            node_ratio_float = float(node_ratio)
            print(f"Using fixed node ratio: {node_ratio_float:.4f}")
            return node_ratio_float



    def compute_coverage(self, selected_nodes):
        '''
        return:
        - selected node 數量 
        - selected node 在原圖的比例 
        - selected node 在現在圖的比例 
        - selected node 的 2hop 在原圖的比例 (original node)
        - selected node 的 2hop 在現在圖的比例 (original node + feature node)
        - selected node 在 2hop edge 的原圖的比例 (node-node)
        - selected node 在 2hop edge 的現在的圖的比例 (node-node + feature-node)
        '''
        # 節點數量
        num_selected_nodes = len(selected_nodes)

        # 節點比例
        node_ratio_ori = num_selected_nodes / self.num_ori_nodes
        node_ratio_current = num_selected_nodes / self.data.x.shape[0]

        # 2-hop 內鄰居
        edge_index = self.data.edge_index.cpu()
        selected_nodes_tensor = torch.tensor(selected_nodes, dtype=torch.long)

        subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
            selected_nodes_tensor, num_hops=2, edge_index=edge_index, relabel_nodes=False
        )

        subset_np = subset.cpu().numpy()
        edge_mask_np = edge_mask.cpu().numpy()

        # 篩出 subset 裡的 original node 數量
        is_original_node_mask = self.data.is_original_node.cpu().numpy()
        num_2hop_original_nodes = np.sum(is_original_node_mask[subset_np])

        # 2-hop 節點 coverage
        node_2hop_ratio_ori = num_2hop_original_nodes / self.num_ori_nodes
        node_2hop_ratio_current = len(subset_np) / self.data.x.shape[0]

        # 2-hop 邊 coverage
        node_node_mask_np = self.data.node_node_mask.cpu().numpy()

        
        node_node_edge_mask_in_2hop = np.logical_and(edge_mask_np, node_node_mask_np)
        num_2hop_original_edge = node_node_edge_mask_in_2hop.sum()

        num_2hop_edges = edge_mask_np.sum()
        edge_2hop_ratio_ori = num_2hop_original_edge / self.num_ori_edges
        edge_2hop_ratio_current = num_2hop_edges / self.data.edge_index.shape[1]

        # 印出 summary
        print(f"Selected {num_selected_nodes} nodes")
        print(f"Node ratio: ori={node_ratio_ori:.4f}, current={node_ratio_current:.4f}")
        print(f"2-hop node ratio: ori={node_2hop_ratio_ori:.4f}, current={node_2hop_ratio_current:.4f}")
        print(f"2-hop edge ratio: ori={edge_2hop_ratio_ori:.4f}, current={edge_2hop_ratio_current:.4f}")

        return (
            num_selected_nodes,
            node_ratio_ori,
            node_ratio_current,
            node_2hop_ratio_ori,
            node_2hop_ratio_current,
            edge_2hop_ratio_ori,
            edge_2hop_ratio_current,
        )


    def _select_stratified_by_degree(self, nodes, num_selected):
        # 計算全 graph 裡每個 node 的 degree
        edge_index = self.data.edge_index.cpu().numpy()
        src_nodes = edge_index[0]
        tgt_nodes = edge_index[1]

        # 計數 degree
        degree_count = {}
        for node in np.concatenate([src_nodes, tgt_nodes]):
            degree_count[node] = degree_count.get(node, 0) + 1

        # 把目標 nodes 裡的 degree 拿出來
        degrees = {node: degree_count.get(node, 0) for node in nodes}

        # 排序 (由小到大 degree)
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1])

        # 分 bin
        bins = np.array_split(sorted_nodes, num_selected)

        # 每個 bin 挑一個
        sampled = [random.choice(list(bin)) for bin in bins if len(bin) > 0]

        # 回傳 node id list
        return [node for node, _ in sampled]
