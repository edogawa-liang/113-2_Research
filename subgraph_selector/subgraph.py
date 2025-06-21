import os
import numpy as np

class CoreSubgraphExtractor:
    def __init__(self, ori_data, remaining_graph, save_dir, dataset, trial_number=None):
        self.ori_data = ori_data
        self.remaining_graph = remaining_graph
        self.save_dir = save_dir
        self.dataset = dataset
        self.trial_number = trial_number

        self.feature_removed_mask = None
        self.edge_removed_mask = None

    def compute_removed_masks(self):
        # print("[CoreSubgraphExtractor] Computing removed feature and edge masks...")

        # === Feature mask ===
        ori_x_np = self.ori_data.x.cpu().numpy()
        remain_x_np = self.remaining_graph.x.cpu().numpy()

        self.feature_removed_mask = ((ori_x_np != 0) & (remain_x_np == 0)).astype(np.int32) # 原本有特徵 vs 現在沒特徵

        # === Edge mask ===
        ori_edge_index_np = self.ori_data.edge_index.cpu().numpy()
        remain_edge_index_np = self.remaining_graph.edge_index.cpu().numpy()

        # print(f"[CoreSubgraphExtractor] ori_edge_index edges: {ori_edge_index_np.shape[1]}")
        # print(f"[CoreSubgraphExtractor] remain_edge_index edges: {remain_edge_index_np.shape[1]}")

        # 先建 set
        ori_edges_set = self._edge_set(ori_edge_index_np)
        remain_edges_set = self._edge_set(remain_edge_index_np)

        # print(f"[CoreSubgraphExtractor] ori_edges_set size: {len(ori_edges_set)}")
        # print(f"[CoreSubgraphExtractor] remain_edges_set size: {len(remain_edges_set)}")

        # 建 map: (src,dst) -> index
        ori_edge_map = {}
        for i in range(ori_edge_index_np.shape[1]):
            src, dst = ori_edge_index_np[0, i], ori_edge_index_np[1, i]
            edge = (src, dst)  # 不做 min/max → 就是直接比對 edge_index 裡的 pair
            ori_edge_map[edge] = i

        num_ori_edges = ori_edge_index_np.shape[1]
        self.edge_removed_mask = np.zeros((1, num_ori_edges), dtype=np.int32)

        removed_edges = ori_edges_set - remain_edges_set
        # print(f"[CoreSubgraphExtractor] removed_edges size: {len(removed_edges)}")

        for edge in removed_edges:
            if edge in ori_edge_map:
                idx = ori_edge_map[edge]
                self.edge_removed_mask[0, idx] = 1

        # print("[CoreSubgraphExtractor] Removed masks computed.")

    def _edge_set(self, edge_index_np):
        edges = set()
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            edge = (src, dst)  # 不做 min/max
            edges.add(edge)
        return edges



    def summary(self):
        print("\n=== [CoreSubgraphExtractor Summary] ===")

        # Feature
        num_nodes, num_features = self.feature_removed_mask.shape
        num_removed_features = self.feature_removed_mask.sum()
        total_features = num_nodes * num_features
        feature_removal_ratio = num_removed_features / total_features * 100

        print(f"Removed features: {num_removed_features} / {total_features} ({feature_removal_ratio:.2f}%)")

        # Edge
        num_edges = self.edge_removed_mask.shape[1]
        num_removed_edges = self.edge_removed_mask.sum()
        edge_removal_ratio = num_removed_edges / num_edges * 100

        print(f"Removed edges: {num_removed_edges} / {num_edges} ({edge_removal_ratio:.2f}%)")

        print("========================================\n")


    def save(self):
        if self.trial_number is None:
            raise ValueError("trial_number is None! Please set trial_number before calling save/load().")

        print("[CoreSubgraphExtractor] Saving removed masks...")

        save_dir = os.path.join("saved", "core_subgraph_mask", self.save_dir, self.dataset)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, f"{self.trial_number}_feature_mask.npy"), self.feature_removed_mask)
        np.save(os.path.join(save_dir, f"{self.trial_number}_edge_mask.npy"), self.edge_removed_mask)
        # 要對照 result 看這個 trial 的子圖做了哪些事情

        print(f"[CoreSubgraphExtractor] Removed masks saved to {save_dir}.")

    def load(self):
        if self.trial_number is None:
            raise ValueError("trial_number is None! Please set trial_number before calling save/load().")

        print("[CoreSubgraphExtractor] Loading removed masks...")

        load_dir = os.path.join("saved", "core_subgraph_mask", self.save_dir, self.dataset)

        self.feature_removed_mask = np.load(os.path.join(load_dir, f"{self.trial_number}_feature_mask.npy"))
        self.edge_removed_mask = np.load(os.path.join(load_dir, f"{self.trial_number}_edge_mask.npy"))

        print(f"[CoreSubgraphExtractor] Removed masks loaded from {load_dir}.")

        return self.feature_removed_mask, self.edge_removed_mask
