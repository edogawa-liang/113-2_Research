import random
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx

class ChooseNodeSelector:
    """
    A class to select nodes from a graph dataset based on different strategies.
    """

    def __init__(self, data, node_ratio="auto", edge_ratio=0.5, strategy="random", manual_nodes=None, mask_type="train"):
        """
        :param data: PyG data object
        :param node_ratio: "auto" for automatic calculation or a numeric value to manually set node selection ratio
        :param edge_ratio: Ensures sufficient edges in the subgraph, required only if node_ratio is 'auto'
        :param strategy: Node selection strategy (e.g., "random", "high_degree", "top_pagerank", "manual", "high_betweenness", "stratified_by_degree", "all")
        :param manual_nodes: Comma-separated list of node indices if using 'manual' strategy
        :param mask_type: Which subset of data to select nodes from ('train', 'test', or 'all')
        """
        self.data = data
        self.node_ratio = node_ratio
        self.edge_ratio = edge_ratio
        self.strategy = strategy
        self.manual_nodes = manual_nodes  # 手動選擇的節點（字串）
        self.mask_type = mask_type
    
    def _calculate_node_ratio(self):
        """
        計算需要選擇的節點數量，以確保解釋子圖能包含指定比例的邊。
        """

        avg_degree = self.data.edge_index.shape[1] / self.data.x.shape[0]
        print(f"Average node degree: {avg_degree}")
        
        if self.node_ratio == "auto":
            target_edges = self.edge_ratio * self.data.edge_index.shape[1]  # 目標邊數
            num_selected_nodes = target_edges / (avg_degree ** 2)  # 需要的節點數
            node_ratio = num_selected_nodes / self.data.x.shape[0]
            print(f"{num_selected_nodes} nodes required to ensure {self.edge_ratio * 100}% edges in the subgraph.")
            print(f"Node ratio: {node_ratio}")
        else:
            node_ratio = float(self.node_ratio)

        return node_ratio

    def _get_node_pool(self):
        if self.mask_type == "train":
            return self.data.train_mask.nonzero().view(-1).tolist()
        elif self.mask_type == "test":
            return self.data.test_mask.nonzero().view(-1).tolist()
        elif self.mask_type == "all":
            return list(range(self.data.x.shape[0]))
        else:
            raise ValueError(f"Unsupported mask_type: {self.mask_type}")

    def select_nodes(self):
        node_pool = self._get_node_pool()

        if self.strategy == "all": # 用在測試集上
            return node_pool
        
        # else 要選擇節點
        node_ratio = self._calculate_node_ratio()
        num_selected = max(1, int(self.data.x.shape[0] * node_ratio))

        if self.strategy == "random":
            random.seed(42)
            return random.sample(node_pool, num_selected)

        elif self.strategy == "high_degree":
            return self._select_high_degree_nodes(node_pool, num_selected)

        elif self.strategy == "top_pagerank":
            return self._select_top_pagerank_nodes(node_pool, num_selected)

        elif self.strategy == "manual":
            return self._select_manual_nodes()

        elif self.strategy == "high_betweenness":
            return self._select_high_betweeness(node_pool, num_selected)
        
        elif self.strategy == "stratified_by_degree":
            return self._select_stratified_by_degree(node_pool, num_selected)
    
        else:
            raise ValueError(f"Unsupported choose_nodes strategy: {self.strategy}")


    def _select_high_degree_nodes(self, nodes, num_selected):
        G = to_networkx(self.data, to_undirected=True)
        G = G.subgraph(nodes)  
        centrality = nx.degree_centrality(G)
        sorted_nodes = sorted(nodes, key=lambda n: centrality.get(n, 0), reverse=True)
        return sorted_nodes[:num_selected]

    def _select_top_pagerank_nodes(self, nodes, num_selected):
        G = to_networkx(self.data, to_undirected=True)
        G = G.subgraph(nodes) 
        pagerank = nx.pagerank(G)
        sorted_nodes = sorted(nodes, key=lambda n: pagerank.get(n, 0), reverse=True)
        return sorted_nodes[:num_selected]

    def _select_manual_nodes(self):
        if not self.manual_nodes:
            raise ValueError("Manual node selection requires a list of node indices.")
        try:
            return [int(n) for n in self.manual_nodes.split(",")]
        except ValueError:
            raise ValueError("Invalid node index format in manual selection. Use comma-separated integers.")

    def _select_high_betweeness(self, nodes, num_selected):
        G = to_networkx(self.data, to_undirected=True)
        G = G.subgraph(nodes)
        betweenness = nx.betweenness_centrality(G)
        sorted_nodes = sorted(nodes, key=lambda n: betweenness.get(n, 0), reverse=True)
        return sorted_nodes[:num_selected]
    
    def _select_stratified_by_degree(self, nodes, num_selected):
        G = to_networkx(self.data, to_undirected=True)
        degrees = dict(G.degree(nodes))
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1])
        bins = np.array_split(sorted_nodes, num_selected)
        sampled = [random.choice(list(bin)) for bin in bins if len(bin) > 0]
        return [node for node, _ in sampled]