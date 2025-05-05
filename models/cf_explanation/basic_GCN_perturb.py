import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import dense_to_sparse
from .utils.utils import create_symm_matrix_from_vec, create_vec_from_symm_matrix


class GCNPerturb(nn.Module):
    """
    GCN + Perturb mask (P_vec)，學習 edge mask，並用 PyG GCNConv forward。
    只允許刪邊 (不允許新增邊)。
    """
    def __init__(self, pyg_gcn_model, sub_adj, beta=0.5):
        super().__init__()
        self.model = pyg_gcn_model
        for param in self.model.parameters():
             param.requires_grad = False # 凍結原模型的参数
               
        self.sub_adj = sub_adj  # dense adjacency matrix
        self.beta = beta

        self.num_nodes = sub_adj.shape[0]
        self.P_vec_size = (self.num_nodes * self.num_nodes - self.num_nodes) // 2 + self.num_nodes

        # 初始化 P_vec
        self.P_vec = Parameter(torch.zeros(self.P_vec_size))  # 全 0 (sigmoid後=0.5)，允許學習刪減
        # self.P_vec = Parameter(torch.empty(self.P_vec_size).uniform_(-0.1, 0.1))
        self.reset_parameters()

    def reset_parameters(self, eps=1e-4):
        with torch.no_grad():
            # 初始化為略小於0 ==> sigmoid後略小於0.5，偏向不保留
            self.P_vec.sub_(eps)

    def get_adj_and_edge_index(self, threshold=False):
        # 生成 symmetric P
        P_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)

        if threshold:
            # binary mask
            P_used = (torch.sigmoid(P_symm) > 0.4999).float()
        else:
            # continuous mask (training 時用)
            P_used = torch.sigmoid(P_symm)
            print("continuous P_used:", P_used)
            
		# perturbed adjacency
        adj = P_used * self.sub_adj # CF-example
        
        # 轉成 edge_index, edge_weight (PyG GCNConv 使用)
        edge_index, edge_weight = dense_to_sparse(adj)

        return adj, edge_index, edge_weight, P_used

    def forward(self, x):
        # 訓練時用 soft mask (可微分)
        _, edge_index, edge_weight, P_used = self.get_adj_and_edge_index(threshold=False)
        # PyG GCNConv forward
        self.model.train()
        return self.model(x, edge_index, edge_weight), P_used

    def forward_prediction(self, x):
        # 推論時用 binary mask (實際效果)
        adj_bin, edge_index_bin, edge_weight_bin, _ = self.get_adj_and_edge_index(threshold=True)
        self.model.eval()
        return self.model(x, edge_index_bin, edge_weight_bin), adj_bin


    def loss(self, output, y_pred_orig, y_pred_new_actual, adj_bin, P_used, alpha=2):
        # output 是 forward 的結果, y_pred_new_actual 是 forward_prediction 的結果
        pred_same = (y_pred_new_actual == y_pred_orig).float()

        output = F.log_softmax(output, dim=0).unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        loss_pred = -F.nll_loss(output, y_pred_orig)

		# 計算 perturbed 與原始的差異
        num_changed = (adj_bin - self.sub_adj).abs().sum().float() / 2
        
		# 計算原始圖的邊數
        num_edges_orig = self.sub_adj.sum().float() / 2
        # print("self.sub_adj", self.sub_adj)

		# 讓目標儘量靠近 0 或 1，而不是靠近 0.5
        loss_reg = (P_used * (1 - P_used)).mean()

        loss_total = pred_same * loss_pred + self.beta * num_changed + alpha * loss_reg

        print(f"原本有幾條邊: {num_edges_orig.item()}, 移除的邊數: {num_changed.item()}")

        return loss_total, num_changed

