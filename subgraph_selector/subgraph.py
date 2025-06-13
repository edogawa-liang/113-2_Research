import os
import numpy as np

class CoreSubgraphExtractor:
    def __init__(self, ori_data, remaining_graph, run_mode, dataset, repeat_id, is_undirected=True):
        self.ori_data = ori_data
        self.remaining_graph = remaining_graph
        self.run_mode = run_mode
        self.dataset = dataset
        self.repeat_id = repeat_id
        self.is_undirected = is_undirected

        self.feature_keep_mask = None
        self.feature_removed_mask = None
        self.edge_keep_mask = None
        self.edge_removed_mask = None

    def compute_masks(self):
        print("[CoreSubgraphExtractor] Computing feature and edge masks...")

        # Feature 部分
        ori_x_np = self.ori_data.x.cpu().numpy()
        remain_x_np = self.remaining_graph.x.cpu().numpy()

        self.feature_keep_mask = (remain_x_np != 0).astype(np.int32)
        self.feature_removed_mask = ((ori_x_np != 0) & (remain_x_np == 0)).astype(np.int32)

        # Edge 部分
        ori_edge_index_np = self.ori_data.edge_index.cpu().numpy()
        remain_edge_index_np = self.remaining_graph.edge_index.cpu().numpy()

        ori_edges_set = self._edge_set(ori_edge_index_np)
        remain_edges_set = self._edge_set(remain_edge_index_np)

        ori_edge_map = {}
        for i in range(ori_edge_index_np.shape[1]):
            src, dst = ori_edge_index_np[0, i], ori_edge_index_np[1, i]
            edge = (min(src, dst), max(src, dst)) if self.is_undirected else (src, dst)
            ori_edge_map[edge] = i

        num_ori_edges = ori_edge_index_np.shape[1]
        self.edge_keep_mask = np.zeros((1, num_ori_edges), dtype=np.int32)
        for edge in remain_edges_set:
            if edge in ori_edge_map:
                self.edge_keep_mask[0, ori_edge_map[edge]] = 1

        self.edge_removed_mask = np.zeros((1, num_ori_edges), dtype=np.int32)
        for edge in ori_edges_set - remain_edges_set:
            if edge in ori_edge_map:
                self.edge_removed_mask[0, ori_edge_map[edge]] = 1

        print("[CoreSubgraphExtractor] Masks computed.")

    def _edge_set(self, edge_index_np):
        edges = set()
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            if self.is_undirected:
                edge = (min(src, dst), max(src, dst))
            else:
                edge = (src, dst)
            edges.add(edge)
        return edges

    def save(self):
        print("[CoreSubgraphExtractor] Saving masks...")

        save_dir = os.path.join("saved", "core_subgraph_mask", self.run_mode, self.dataset)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, f"repeat{self.repeat_id}_feature_keep_mask.npy"), self.feature_keep_mask)
        np.save(os.path.join(save_dir, f"repeat{self.repeat_id}_feature_removed_mask.npy"), self.feature_removed_mask)
        np.save(os.path.join(save_dir, f"repeat{self.repeat_id}_edge_keep_mask.npy"), self.edge_keep_mask)
        np.save(os.path.join(save_dir, f"repeat{self.repeat_id}_edge_removed_mask.npy"), self.edge_removed_mask)

        print(f"[CoreSubgraphExtractor] Masks saved to {save_dir}.")

    def load(self):
        print("[CoreSubgraphExtractor] Loading masks...")

        load_dir = os.path.join("saved", "core_subgraph_mask", self.run_mode, self.dataset)

        self.feature_keep_mask = np.load(os.path.join(load_dir, f"repeat{self.repeat_id}_feature_keep_mask.npy"))
        self.feature_removed_mask = np.load(os.path.join(load_dir, f"repeat{self.repeat_id}_feature_removed_mask.npy"))
        self.edge_keep_mask = np.load(os.path.join(load_dir, f"repeat{self.repeat_id}_edge_keep_mask.npy"))
        self.edge_removed_mask = np.load(os.path.join(load_dir, f"repeat{self.repeat_id}_edge_removed_mask.npy"))

        print(f"[CoreSubgraphExtractor] Masks loaded from {load_dir}.")

        return self.feature_keep_mask, self.feature_removed_mask, self.edge_keep_mask, self.edge_removed_mask
