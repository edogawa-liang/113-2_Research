import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .basic_GCN_perturb import GCNPerturb
from .utils.utils import get_degree_matrix
import matplotlib.pyplot as plt
from utils.device import DEVICE



class CFExplainer:
    """
    Counterfactual Explainer using perturbable adjacency and original PyG model.
    """

    def __init__(self, model, sub_adj, sub_feat, sub_labels, y_pred_orig, beta, device):
        super(CFExplainer, self).__init__()
        

        self.model = model.to(device) # PyG GCNConv based model
        self.model.eval()
        self.device = DEVICE
        self.sub_adj = sub_adj.to(self.device)
        self.sub_feat = sub_feat.to(self.device)
        self.sub_labels = sub_labels.to(self.device)
        self.y_pred_orig = y_pred_orig.to(self.device)
        
        self.beta = beta
        self.loss_total_list = []

        # Create perturbed version of the model
        self.cf_model = GCNPerturb(model, sub_adj, beta=beta)
        self.best_cf_adj = None  # 儲存最佳 perturb 結果

        print("\n[CF model parameters]")
        for name, param in self.cf_model.named_parameters():
            print(f"{name:25s} shape: {tuple(param.shape)} requires_grad: {param.requires_grad}")


    def explain(self, cf_optimizer, node_idx, new_idx, lr, n_momentum, num_epochs, node_dict):
        self.node_idx = node_idx
        self.new_idx = new_idx

        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum)
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)

        best_loss = np.inf
        P_used_best = None

        for epoch in range(num_epochs):
            print("Epoch: ", epoch)
            loss_total, flipped, cf_adj_bin, P_used = self.train()
            self.loss_total_list.append(loss_total)

            if flipped and loss_total < best_loss:
                self.best_cf_adj = cf_adj_bin.detach().clone()
                P_used_best = P_used.detach().clone()
                best_loss = loss_total
		
        if self.best_cf_adj is None:
            print("No counterfactual example found.")
            return None
        
        # 產生 removed edges
        removed_edges_global, removed_edge_pairs_sub = self.get_removed_edges_from_original_index(node_dict)

        # 取得邊的重要度
        edge_importance = self.get_removed_edges_importance(P_used_best, removed_edge_pairs_sub)

        # 確保 removed_edges_global, edge_importance 長度一致
        assert removed_edges_global.shape[1] == edge_importance.shape[0], f"邊的數量與重要度數量不一致，邊數: {removed_edges_global.shape[1]}, 重要度數量: {edge_importance.shape[0]}"

            

        # 回傳結果
        cf_info = {
            "node_idx": node_idx,
            "cf_explanation": removed_edges_global.cpu(),
            "edge_importance": edge_importance.cpu(),
        }

        return cf_info

    def train(self):
        self.cf_model.train()
        self.cf_optimizer.zero_grad()

        # Binary forward (real prediction)
        output_actual, self.cf_adj = self.cf_model.forward_prediction(self.sub_feat)
        
		# Differentiable forward (soft mask)
        # P_used 邊被保留下來的機率 (<0.5 的就是解釋)
        output, P_used = self.cf_model.forward(self.sub_feat)
        print("output:", output[self.new_idx])

        # Predictions
        # y_pred_new = torch.argmax(output[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])
        flipped = y_pred_new_actual != self.y_pred_orig

        print("y_pred_new_actual:", y_pred_new_actual, "y_pred_orig:", self.y_pred_orig)

        # Calculate loss
        loss_total, num_changed = self.cf_model.loss(output[self.new_idx], self.y_pred_orig, y_pred_new_actual, self.cf_adj, P_used)
        loss_total.backward() 
        self.cf_optimizer.step()
        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        
        print("loss_total:", loss_total.item(), "num_changed:", num_changed.item())
        
        return loss_total.item(), flipped, self.cf_adj, P_used


    def get_removed_edges_from_original_index(self, node_dict):
        """
        回傳 perturb 後移除的邊（edge_index 格式，原圖對應編號）
        """
        reverse_node_dict = {v: k for k, v in node_dict.items()}
        edge_mask = (self.sub_adj - self.best_cf_adj.to(self.sub_adj.device)) > 0
        removed_edge_pairs_sub = edge_mask.nonzero(as_tuple=False)  # [num_removed_edges, 2]

        removed_edges_global = torch.tensor([
            [reverse_node_dict[i.item()], reverse_node_dict[j.item()]]
            for i, j in removed_edge_pairs_sub
        ], device=self.device).T  # [2, num_edges]

        return removed_edges_global, removed_edge_pairs_sub
    

    def get_removed_edges_importance(self, P_used, removed_edge_pairs_sub):
        importance_matrix = 1.0 - P_used
        removed_importance = torch.tensor([
            importance_matrix[i.item(), j.item()]
            for i, j in removed_edge_pairs_sub
        ], device=self.device)

        print("Removed importance:", removed_importance)
        
        return removed_importance



    def plot_loss(self, save_path):
        """
        繪製每個 epoch 的 loss 變化
        """
        if not self.loss_total_list:
            print("No loss recorded.")
            return

        # 確保 loss_plot 資料夾存在
        save_dir = os.path.join(os.path.dirname(save_path), "loss_plot")
        os.makedirs(save_dir, exist_ok=True)
        full_save_path = os.path.join(save_dir, os.path.basename(save_path))    
        
        plt.figure(figsize=(6, 4))
        plt.plot(self.loss_total_list, linestyle='-')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(full_save_path)
        plt.close()
        print(f"Loss plot saved to {full_save_path}")