# struct_initializers.py

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix

class StructureFeatures:
    """
    Base class：負責
    1. 計算並儲存每個節點的 degree、PageRank、Betweenness（未正規化值）
    2. 對 degree、PageRank、Betweenness 做 Min–Max 正規化
    3. 提供查詢「每個節點的正規化指標」接口
    """

    def __init__(self, G: nx.Graph):
        """
        G: networkx.Graph（節點應確保 0..n-1 的映射，若不是連號請先手動 relabel）
        """
        self.G = G
        self.nodes = list(G.nodes())  # 假設 node IDs = 0,1,2,...,n-1；若不是，請事先 relabel
        self.n = len(self.nodes)

        # 1) 計算原始度數、PageRank、Betweenness
        #    注意：Betweenness 在大型圖上可能耗時，可視情況使用近似版本
        deg_dict = dict(G.degree())
        pr_dict = nx.pagerank(G, alpha=0.85, max_iter=100)
        bet_dict = nx.betweenness_centrality(G)  # 精確版，若太慢可改用 k-sample 約略

        # 2) 把 dict 轉成與節點索引對應的 list
        deg_list = np.array([deg_dict[v] for v in self.nodes], dtype=float)
        pr_list  = np.array([pr_dict[v]  for v in self.nodes], dtype=float)
        bet_list = np.array([bet_dict[v] for v in self.nodes], dtype=float)

        # 3) Min–Max 正規化（若最大值==最小值，皆設為 0）
        self.deg_norm  = self._min_max_normalize(deg_list)
        self.pr_norm   = self._min_max_normalize(pr_list)
        self.bet_norm  = self._min_max_normalize(bet_list)

        # 4) 建立形狀 (n, 3) 的結構指標矩陣（順序：deg, pr, bet）
        self.struct_mat = np.vstack([self.deg_norm, self.pr_norm, self.bet_norm]).T  # shape=(n,3)

    def _min_max_normalize(self, arr: np.ndarray) -> np.ndarray:
        vmin, vmax = arr.min(), arr.max()
        if vmax - vmin < 1e-12:
            return np.zeros_like(arr)
        return (arr - vmin) / (vmax - vmin)

    def get_struct_matrix(self) -> np.ndarray:
        """
        回傳 (n,3) 的結構特徵矩陣（正規化後）。
        columns 為 [degree_norm, pagerank_norm, betweenness_norm]
        """
        return self.struct_mat.copy()


# -----------------------------------------------------------------------------
# 以下定義七種初始化策略的 Class，每個 Class 都繼承自 StructureFeatures，
# 以便直接使用 'self.struct_mat' 來取得 3 維結構指標。
# -----------------------------------------------------------------------------

class AllOne(StructureFeatures):
    """
    初始化方法一：全 1
    - 只回傳常數 1 這個維度，形狀 = (n, 1)
    - 模型必須完全靠 GNN 多層訊息傳遞才能從 degree … 等推算結構
    """

    def __init__(self, G: nx.Graph):
        super().__init__(G)

    def get_features(self) -> np.ndarray:
        """
        回傳 (n,1)，全 1
        """
        return np.ones((self.n, 1), dtype=float)


class OnePlusStruct(StructureFeatures):
    """
    初始化方法二：1 + [PageRank, Betweenness, Degree]
    - 回傳 (n,4)，第一維恆為 1，其餘 3 維分別是 deg_norm, pr_norm, bet_norm
    """

    def __init__(self, G: nx.Graph):
        super().__init__(G)

    def get_features(self) -> np.ndarray:
        """
        回傳 (n,4)：
        [ [1, deg_norm(v), pr_norm(v), bet_norm(v)]  for v in 0..n-1 ]
        """
        ones = np.ones((self.n, 1), dtype=float)
        return np.concatenate([ones, self.struct_mat], axis=1)  # shape=(n,4)


class OneHot(StructureFeatures):
    """
    初始化方法三：One Hot
    - 回傳 (n,n) 的稀疏或稠密 One-Hot 向量。這會非常耗費記憶體，僅在 n<=5000 建議使用。
    """

    def __init__(self, G: nx.Graph):
        super().__init__(G)

    def get_features(self, sparse: bool = False):
        """
        回傳 (n,n) 的 One-Hot 矩陣
        若 sparse=True，則回傳 scipy.sparse.coo_matrix
        若 sparse=False，回傳 dense NumPy ndarray
        """
        if sparse:
            # 使用 COO 格式建立稀疏 One-Hot
            row_idx = np.arange(self.n)
            col_idx = np.arange(self.n)
            data    = np.ones(self.n, dtype=float)
            return coo_matrix((data, (row_idx, col_idx)), shape=(self.n, self.n))
        else:
            # 直接建立 dense 一維矩陣
            eye = np.eye(self.n, dtype=float)  # shape=(n,n)
            return eye


class OneHotMLP(StructureFeatures):
    """
    初始化方法四：One Hot → MLP Embedding
    - 使用 PyTorch 的 nn.Embedding 做「稀疏 One-Hot → d_embed 維度」的映射
    - 最後回傳一個 torch.Tensor, shape=(n, d_embed)
    """

    def __init__(self, G: nx.Graph, d_embed: int = 32):
        """
        d_embed: 嵌入維度（One hot 壓縮後的最終維度）
        """
        super().__init__(G)
        self.d_embed = d_embed
        # 建立 nn.Embedding：本質相當於將稀疏 n 維 One-Hot 投射到 d_embed 維
        self.embedding = nn.Embedding(num_embeddings=self.n, embedding_dim=self.d_embed)

    def get_features(self) -> torch.Tensor:
        """
        回傳 (n, d_embed) 的 torch.Tensor，每一列是節點 v 的嵌入向量
        其內部相當於：One-Hot[v] × W  → 最終向量
        """
        # 建立 node indices Tensor: [0,1,2,...,n-1]
        idx = torch.arange(self.n, dtype=torch.long)
        # embedding 查詢
        return self.embedding(idx)  # shape=(n, d_embed)


class OneHotMLPPlusStruct(StructureFeatures):
    """
    初始化方法五：One Hot → MLP Embedding + [PageRank, Betweenness, Degree]
    - 先透過 nn.Embedding 把 One-Hot 壓到 d_embed
    - 再把 結構指標 (3 維) concat 到 嵌入後 → 最終 (n, d_embed+3)
    """

    def __init__(self, G: nx.Graph, d_embed: int = 32):
        super().__init__(G)
        self.d_embed = d_embed
        self.embedding = nn.Embedding(num_embeddings=self.n, embedding_dim=self.d_embed)

    def get_features(self) -> torch.Tensor:
        """
        回傳 (n, d_embed+3) 的 torch.Tensor：
        [ embedding(v), deg_norm(v), pr_norm(v), bet_norm(v) ]
        """
        # (1) Embedding 查詢
        idx = torch.arange(self.n, dtype=torch.long)
        x_embed = self.embedding(idx)  # shape=(n, d_embed)

        # (2) 將 NumPy 的 struct_mat 轉成 torch.Tensor
        struct_tensor = torch.from_numpy(self.struct_mat).to(dtype=torch.float32)  # (n,3)

        # (3) concat
        return torch.cat([x_embed, struct_tensor], dim=1)  # shape=(n, d_embed+3)


class RandomInit(StructureFeatures):
    """
    初始化方法六：隨機初始化
    - 為每個節點分配一個固定的隨機向量 (dim=d_rand)
    - 最好把這個向量 set 為可訓練 (nn.Parameter) 或凍結隨機不更新都可
    """

    def __init__(self, G: nx.Graph, d_rand: int = 32, trainable: bool = False):
        """
        d_rand: 隨機向量維度
        trainable: 若 True，隨機向量將設為 nn.Parameter，訓練期間可被微調
        """
        super().__init__(G)
        self.d_rand = d_rand
        # 產生隨機矩陣 (n, d_rand)
        rand_mat = torch.randn((self.n, self.d_rand), dtype=torch.float32)
        if trainable:
            self.rand_embed = nn.Parameter(rand_mat)  # 可訓練參數
        else:
            self.rand_embed = torch.nn.Parameter(rand_mat, requires_grad=False)  # 不更新

    def get_features(self) -> torch.Tensor:
        """
        回傳 (n, d_rand) 的 torch.Tensor 隨機矩陣
        """
        return self.rand_embed  # shape=(n, d_rand)


class RandomInitPlusStruct(StructureFeatures):
    """
    初始化方法七：隨機初始化 + [PageRank, Betweenness, Degree]
    - 為每個節點分配一個固定的隨機向量 (dim=d_rand)，
      並把結構指標 (3 維) concat → 最終 (n, d_rand+3)
    """

    def __init__(self, G: nx.Graph, d_rand: int = 32, trainable: bool = False):
        super().__init__(G)
        self.d_rand = d_rand
        rand_mat = torch.randn((self.n, self.d_rand), dtype=torch.float32)
        if trainable:
            self.rand_embed = nn.Parameter(rand_mat)
        else:
            self.rand_embed = torch.nn.Parameter(rand_mat, requires_grad=False)

    def get_features(self) -> torch.Tensor:
        """
        回傳 (n, d_rand+3) 的 torch.Tensor：
        [ rand_vec(v), deg_norm(v), pr_norm(v), bet_norm(v) ]
        """
        # (1) 隨機部分
        rand_tensor = self.rand_embed  # shape=(n, d_rand)

        # (2) 結構指標部分
        struct_tensor = torch.from_numpy(self.struct_mat).to(dtype=torch.float32)  # (n,3)

        # (3) concat
        return torch.cat([rand_tensor, struct_tensor], dim=1)  # shape=(n, d_rand+3)


# -----------------------------------------------------------------------------
# Usage Example (寫在檔案外或測試檔):
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 範例：讀入 Zachary's Karate Club 這張小圖，節點數=34
    G = nx.karate_club_graph()
    # 重新標號為 0..33（若 networkx 生成的節點已經是 0..33，可略過）
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering="default")

    # 1) 全 1
    all1 = AllOne(G)
    feat_all1 = all1.get_features()
    print("AllOne:", feat_all1.shape)  # (34,1)

    # 2) 1 + [PR, Bet, Deg]
    plus_struct = OnePlusStruct(G)
    feat_plus = plus_struct.get_features()
    print("OnePlusStruct:", feat_plus.shape)  # (34,4)

    # 3) One Hot
    onehot = OneHot(G)
    feat_oh_dense = onehot.get_features(sparse=False)
    print("OneHot Dense:", feat_oh_dense.shape)  # (34,34)
    feat_oh_sparse = onehot.get_features(sparse=True)
    print("OneHot Sparse:", feat_oh_sparse.shape)  # (34,34) sparse

    # 4) One Hot → MLP Embedding (d_embed=16)
    oh_mlp = OneHotMLP(G, d_embed=16)
    feat_oh_mlp = oh_mlp.get_features()
    print("OneHotMLP:", feat_oh_mlp.shape)  # (34,16)

    # 5) One Hot → MLP Embedding + [PR, Bet, Deg]
    oh_mlp_plus = OneHotMLPPlusStruct(G, d_embed=16)
    feat_oh_mlp_plus = oh_mlp_plus.get_features()
    print("OneHotMLPPlusStruct:", feat_oh_mlp_plus.shape)  # (34,19)

    # 6) Random Init (d_rand=16, 不訓練)
    rand_init = RandomInit(G, d_rand=16, trainable=False)
    feat_rand = rand_init.get_features()
    print("RandomInit:", feat_rand.shape)  # (34,16)

    # 7) Random Init + [PR, Bet, Deg]
    rand_plus = RandomInitPlusStruct(G, d_rand=16, trainable=False)
    feat_rand_plus = rand_plus.get_features()
    print("RandomInitPlusStruct:", feat_rand_plus.shape)  # (34,19)

    # 你可以把上面各種 feat_* 丟到自己的 GNN 模型裡做進一步訓練。
