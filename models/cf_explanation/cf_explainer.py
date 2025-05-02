# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .utils.utils import get_degree_matrix
from .basic_GCN_perturb import GCNSyntheticPerturb
from .utils.utils import normalize_adj


class CFExplainer:
	"""
	CF Explainer class, returns counterfactual subgraph
	"""
	def __init__(self, model, sub_adj, sub_feat, n_hid, dropout,
	              sub_labels, y_pred_orig, num_classes, beta, device):
		super(CFExplainer, self).__init__()
		self.model = model
		self.model.eval()
		self.sub_adj = sub_adj
		self.sub_feat = sub_feat
		self.n_hid = n_hid
		self.dropout = dropout
		self.sub_labels = sub_labels
		self.y_pred_orig = y_pred_orig
		self.beta = beta
		self.num_classes = num_classes
		self.device = device

		# Instantiate CF model class, load weights from original model
		self.cf_model = GCNSyntheticPerturb(self.sub_feat.shape[1], n_hid, n_hid,
		                                    self.num_classes, self.sub_adj, dropout, beta)

		# 問題出在這? cf是自訂 forward, original model是torch_geometric的forward
		self.cf_model.gc1.weight.data = self.model.conv1.lin.weight.data.T.clone()
		self.cf_model.gc1.bias.data = self.model.conv1.bias.data.clone()

		self.cf_model.gc2.weight.data = self.model.conv2.lin.weight.data.T.clone()
		self.cf_model.gc2.bias.data = self.model.conv2.bias.data.clone()
		
		print("model", self.model)
		print("cf", self.cf_model)
		self.cf_model.load_state_dict(self.model.state_dict(), strict=False)
		# print("model dict", self.model.state_dict())
		# print("cf model dict", self.cf_model.state_dict())

		print("\n[Original model parameters]")
		for name, param in self.model.state_dict().items():
			print(f"{name:25s} shape: {tuple(param.shape)}")

		print("\n[CF model parameters]")
		for name, param in self.cf_model.state_dict().items():
			print(f"{name:25s} shape: {tuple(param.shape)}")

		# Freeze weights from original model in cf_model
		for name, param in self.cf_model.named_parameters():
			if name.endswith("weight") or name.endswith("bias"):
				param.requires_grad = False
		for name, param in self.model.named_parameters():
			print("orig model requires_grad: ", name, param.requires_grad)
			#  在 CF explainer 裡，不是用 self.model 的 optimizer，cf model 的 weight 也沒有要做更新，只學習 P_vec
		for name, param in self.cf_model.named_parameters():
			print("cf model requires_grad: ", name, param.requires_grad)



	def explain(self, cf_optimizer, node_idx, new_idx, lr, n_momentum, num_epochs):
		self.node_idx = node_idx
		self.new_idx = new_idx

		self.x = self.sub_feat
		self.A_x = self.sub_adj
		self.D_x = get_degree_matrix(self.A_x)

		if cf_optimizer == "SGD" and n_momentum == 0.0:
			self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
		elif cf_optimizer == "SGD" and n_momentum != 0.0:
			self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum)
		elif cf_optimizer == "Adadelta":
			self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)


		best_loss = np.inf
		num_cf_examples = 0
		best_cf_example = []

		for epoch in range(num_epochs):
			print("Epoch: ", epoch)
			new_example, loss_total = self.train(epoch)
			if new_example != [] and loss_total < best_loss:
				best_cf_example = new_example
				self.best_cf_adj = self.cf_adj.detach().clone()
				best_loss = loss_total
				num_cf_examples += 1 # 紀錄成功產出的 CF 數量
		# print("{} CF examples for node_idx = {}".format(num_cf_examples, self.node_idx))
		# print(" ")
		print("Best CF example: ", best_cf_example)
		return best_cf_example


	def train(self, epoch):
		t = time.time()
		self.cf_model.train()
		self.cf_optimizer.zero_grad()

		# output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
		# output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction

		output = self.cf_model.forward(self.x, self.A_x) # 用連續的 mask P_hat 得到的結果，是可微分的 prediction，用來計算 loss
		output_actual, self.P = self.cf_model.forward_prediction(self.x) # 用離散的 mask P 產生，實際上會產生效果的預測
		# print("output: ", output[self.new_idx], "output_actual: ", output_actual[self.new_idx])
	
		# Need to use new_idx from now on since sub_adj is reindexed
		y_pred_new = torch.argmax(output[self.new_idx])  # 解釋模型下的預測
		y_pred_new_actual = torch.argmax(output_actual[self.new_idx]) # binary mask 下的預測，可檢查是否真的 flip 預測結果
		print("y_pred_new_actual: ", y_pred_new_actual, "y_pred_orig: ", self.y_pred_orig)

		# 在 training loop 中，optimizer.step() 之前加上以下檢查：
		with torch.no_grad():
			P_before = self.cf_model.P_vec.clone()
			
		# loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
		loss_total, loss_pred, loss_graph_dist, self.cf_adj = self.cf_model.loss(output[self.new_idx], self.y_pred_orig, y_pred_new_actual)
		loss_total.backward()
		
		# add
		# print("P_vec.grad", self.cf_model.P_vec.grad)

		clip_grad_norm_(self.cf_model.parameters(), 2.0)
		self.cf_optimizer.step()

		# optimizer.step() 之後再比較 P_vec
		with torch.no_grad():
			P_after = self.cf_model.P_vec.clone()
			change = torch.sum(torch.abs(P_before - P_after)).item()
		print(f'P_vec changed by total: {change:.6f}')
		print("P_vec requires_grad:", self.cf_model.P_vec.requires_grad)  # 應該是 True

		# print('Node idx: {}'.format(self.node_idx),
		#       'New idx: {}'.format(self.new_idx),
		# 	  'Epoch: {:04d}'.format(epoch + 1),
		#       'loss: {:.4f}'.format(loss_total.item()),
		#       'pred loss: {:.4f}'.format(loss_pred.item()),
		#       'graph loss: {:.4f}'.format(loss_graph_dist.item()))
		# print('Output: {}\n'.format(output[self.new_idx].data),
		#       'Output nondiff: {}\n'.format(output_actual[self.new_idx].data),
		#       'orig pred: {}, new pred: {}, new pred nondiff: {}'.format(self.y_pred_orig, y_pred_new, y_pred_new_actual))
		# print(" ")
		cf_stats = []
		if y_pred_new_actual != self.y_pred_orig:
			cf_stats = [self.node_idx, self.new_idx,
			            self.cf_adj.detach().cpu().numpy(), self.sub_adj.detach().cpu().numpy(),
			            self.y_pred_orig.item(), y_pred_new.item(),
			            y_pred_new_actual.item(), self.sub_labels[self.new_idx].cpu().numpy(),
			            self.sub_adj.shape[0], loss_total.item(),
			            loss_pred.item(), loss_graph_dist.item()]


		return(cf_stats, loss_total.item())
	

	# 額外加的 (還沒驗證)
	def get_removed_edges_from_original_index(self, node_dict):
		"""
		回傳原圖節點編號與從原圖中被 perturb 掉的邊（edge_index 格式）

		Args:
			node_dict: dict，原圖 -> 子圖的節點編號對應

		Returns:
			dict:
				- "target_node": 原圖節點編號
				- "removed_edges": tensor [2, num_edges]，原圖中的 edge_index 表示
		"""
		reverse_node_dict = {v: k for k, v in node_dict.items()}
		edge_mask = (self.sub_adj - self.best_cf_adj.to(self.sub_adj.device)) > 0 # 最好的那次CF
		removed_edge_indices = edge_mask.nonzero(as_tuple=False)

		removed_edges_global = torch.tensor([
			[reverse_node_dict[i.item()], reverse_node_dict[j.item()]]
			for i, j in removed_edge_indices
		]).T  # shape: [2, num_edges]

		return removed_edges_global

