import os
import errno
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph
from torch_geometric.utils import to_undirected, is_undirected



def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise


def safe_open(path, w):
	''' Open "path" for writing, creating any parent directories as needed.'''
	mkdir_p(os.path.dirname(path))
	return open(path, w)


def accuracy(output, labels):
	preds = output.max(1)[1].type_as(labels)
	correct = preds.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)


def get_degree_matrix(adj):
	return torch.diag(sum(adj))


def normalize_adj(adj):
	# Normalize adjacancy matrix according to reparam trick in GCN paper
	A_tilde = adj + torch.eye(adj.shape[0])
	D_tilde = get_degree_matrix(A_tilde)
	# Raise to power -1/2, set all infs to 0s
	D_tilde_exp = D_tilde ** (-1 / 2)
	D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

	# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
	norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
	return norm_adj


def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
    """
    取得子圖，並根據原圖是否為 undirected，自動決定是否將子圖轉為 undirected。
    """

    # 先判斷原圖是不是 undirected
    original_is_undirected = is_undirected(edge_index)

    # 擷取 k-hop 子圖（這時可能 directed 也可能 undirected，看 edge_index 來決定）
    subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx, n_hops, edge_index, relabel_nodes=True)

    # 如果原圖是 undirected，就確保子圖也是 undirected
    if original_is_undirected:
        sub_edge_index = to_undirected(sub_edge_index)

    sub_feat = features[subset]
    sub_labels = labels[subset]
    # subset = subset.cpu()  # 先搬回 CPU
    node_dict = {int(orig): i for i, orig in enumerate(subset.tolist())}

    return sub_edge_index, sub_feat, sub_labels, node_dict



# Original CFExplainer code
# def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
# 	edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index[0])     # Get all nodes involved
# 	edge_subset_relabel = subgraph(edge_subset[0], edge_index[0], relabel_nodes=True)       # Get relabelled subset of edges
# 	sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
# 	sub_feat = features[edge_subset[0], :]
# 	sub_labels = labels[edge_subset[0]]
# 	new_index = np.array([i for i in range(len(edge_subset[0]))])
# 	node_dict = dict(zip(edge_subset[0].numpy(), new_index))        # Maps orig labels to new
# 	# print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
# 	return sub_adj, sub_feat, sub_labels, node_dict

def create_symm_matrix_from_vec(vector, size):
    device = vector.device  # <-- 讓 matrix 跟 vector 一樣的 device
    matrix = torch.zeros(size, size, device=device)
    idx = torch.triu_indices(size, size, offset=1, device=device)  # <-- 讓 idx 也在相同 device
    matrix[idx[0], idx[1]] = vector
    matrix[idx[1], idx[0]] = vector
    return matrix


def create_vec_from_symm_matrix(matrix, P_vec_size):
	idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
	vector = matrix[idx[0], idx[1]]
	return vector


def index_to_mask(index, size):
	mask = torch.zeros(size, dtype=torch.bool, device=index.device)
	mask[index] = 1
	return mask

def get_S_values(pickled_results, header):
	df_prep = []
	for example in pickled_results:
		if example != []:
			df_prep.append(example[0])
	return pd.DataFrame(df_prep, columns=header)


def redo_dataset_pgexplainer_format(dataset, train_idx, test_idx):

	dataset.data.train_mask = index_to_mask(train_idx, size=dataset.data.num_nodes)
	dataset.data.test_mask = index_to_mask(test_idx[len(test_idx)], size=dataset.data.num_nodes)