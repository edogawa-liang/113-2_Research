import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .utils.utils import get_degree_matrix, normalize_adj, create_symm_matrix_from_vec, create_vec_from_symm_matrix


# H = A @ (X @ W) + b (自由地輸入任意的 adjacency)
class GraphConvolutionPerturb(nn.Module):
	"""
	Similar to GraphConvolution except includes P_hat
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolutionPerturb, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias is not None:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)


	def forward(self, input, adj):
		support = torch.mm(input, self.weight)
		output = torch.spmm(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
		       + str(self.in_features) + ' -> ' \
		       + str(self.out_features) + ')'



class GCNSyntheticPerturb(nn.Module):
	"""
	2-layer GCN used in GNN Explainer synthetic tasks
	"""
	def __init__(self, nfeat, nhid, nhid2, nclass, adj, dropout, beta, edge_additions=False):
		super(GCNSyntheticPerturb, self).__init__()
		self.adj = adj
		self.nclass = nclass
		self.beta = beta
		print("self.adj", self.adj)
		self.num_nodes = self.adj.shape[0]
		self.edge_additions = edge_additions      # are edge additions included in perturbed matrix

		# P_hat needs to be symmetric ==> learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
		self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2)  + self.num_nodes

		if self.edge_additions:
			self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size))) # 新增擾亂邊
		else:
			# self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size))) # 刪除邊 (Ori CF)

			self.P_vec = Parameter(torch.FloatTensor(self.P_vec_size).fill_(0.0))
			# self.P_vec = Parameter(torch.empty(self.P_vec_size).uniform_(-0.1, 0.1))
			# self.P_vec = Parameter(torch.full((self.P_vec_size,), -0.001)) # sigmoid後不到但接近，即所有邊都移除
			# self.P_vec = Parameter(torch.full((self.P_vec_size,), 0.001)) # sigmoid後比0.5大一點點，但很容易變成小於0.5並移除邊

		self.reset_parameters()

		self.gc1 = GraphConvolutionPerturb(nfeat, nhid)
		self.gc2 = GraphConvolutionPerturb(nhid, nclass)
		self.dropout = dropout

	def reset_parameters(self, eps=10**-4):
		# Think more about how to initialize this
		with torch.no_grad():
			if self.edge_additions:
				adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size).numpy()
				for i in range(len(adj_vec)):
					if i < 1:
						adj_vec[i] = adj_vec[i] - eps
					else:
						adj_vec[i] = adj_vec[i] + eps
				torch.add(self.P_vec, torch.FloatTensor(adj_vec))       #self.P_vec is all 0s
			else:
				torch.sub(self.P_vec, eps) # 意圖是讓 sigmoid(P_vec) 稍微小於 0.73，接近 0.5，有機會「不保留原來的邊」



	def forward(self, x, sub_adj):
		self.sub_adj = sub_adj
		# Same as normalize_adj in utils.py except includes P_hat in A_tilde
		self.sub_adj = sub_adj.to(self.P_vec.device)
		self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)      # Ensure symmetry

		A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
		A_tilde.requires_grad = True

		print("self.P_hat_symm after sigmoid", F.sigmoid(self.P_hat_symm))
		if self.edge_additions:         # Learn new adj matrix directly
			A_tilde = F.sigmoid(self.P_hat_symm) + torch.eye(self.num_nodes, device=self.P_hat_symm.device)  # Use sigmoid to bound P_hat in [0,1]
			
		else:       # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
			A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(self.num_nodes, device=self.P_hat_symm.device)       # Use sigmoid to bound P_hat in [0,1]

		D_tilde = get_degree_matrix(A_tilde).detach() + 1e-6      # Don't need gradient of this
		# Raise to power -1/2, set all infs to 0s
		D_tilde_exp = D_tilde ** (-1 / 2)
		D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

		# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
		norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
		norm_adj = norm_adj.to(x.device)

		# print('x shape initially', x.shape)
		x = F.relu(self.gc1(x, norm_adj))
		# print('x.shape after gc1: ', x.shape)
		x = F.dropout(x, self.dropout, training=self.training)
		x = self.gc2(x, norm_adj)
		# print('x.shape after gc2: ', x.shape)
		# print("forward x", x)
		x = x / torch.clamp(x.abs().max(), min=1e-6) # add
		return F.log_softmax(x, dim=1)


	def forward_prediction(self, x): 
		# 只是用來確定預測是否改變，不是用他計算loss
		# Same as forward but uses P instead of P_hat ==> non-differentiable
		# but needed for actual predictions

		# print("self.P_hat_symm")
		# print(self.P_hat_symm)
		self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()      # threshold P_hat
		self.adj = self.adj.to(self.P.device)
		print("self.P", self.P)

		if self.edge_additions:
			A_tilde = self.P + torch.eye(self.num_nodes)
		else:
			A_tilde = self.P * self.adj + torch.eye(self.num_nodes, device=self.P.device)

		D_tilde = get_degree_matrix(A_tilde) + 1e-6
		# Raise to power -1/2, set all infs to 0s
		D_tilde_exp = D_tilde ** (-1 / 2)
		D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

		# print("A_tilde", A_tilde)
		# print("D_tilde", D_tilde)
		# print("D_tilde_exp", D_tilde_exp)

		# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
		norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
		norm_adj = norm_adj.to(x.device)

		x = F.relu(self.gc1(x, norm_adj))
		x = F.dropout(x, self.dropout, training=self.training)
		x = self.gc2(x, norm_adj)
		return F.log_softmax(x, dim=1), self.P


	def loss(self, output, y_pred_orig, y_pred_new_actual):
		pred_same = (y_pred_new_actual == y_pred_orig).float() # 0 or 1, 1 才會計算到loss
		# print("pred_same: ", pred_same)

		# Need dim >=2 for F.nll_loss to work
		output = output.unsqueeze(0)
		print('output', output)	
		y_pred_orig = y_pred_orig.unsqueeze(0)

		if self.edge_additions:
			cf_adj = self.P
		else:
			cf_adj = self.P * self.adj
		cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient

		# Calculate the k-hop edge from this node
		# num_edges = torch.sum(self.adj == 1).item() // 2
		# print("number of edges from node: ", num_edges)
		
		# Want negative in front to maximize loss instead of minimizing it to find CFs
		loss_pred = - F.nll_loss(output, y_pred_orig)
		print('loss_pred', loss_pred)
		loss_graph_dist = sum(sum(abs(cf_adj - self.adj))) / 2      # Number of edges changed (symmetrical)
		print('Number of edges changed: ', loss_graph_dist)
		# Zero-out loss_pred with pred_same if prediction flips
		loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
		print('total loss', loss_total)
		return loss_total, loss_pred, loss_graph_dist, cf_adj
