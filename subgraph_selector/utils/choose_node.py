import random
import networkx as nx
from torch_geometric.utils import to_networkx

class ChooseNodeSelector:
    """
    A class to select nodes from a graph dataset based on different strategies.
    """

    def __init__(self, data, node_ratio="auto", edge_ratio=0.5, strategy="random", manual_nodes=None):
        """
        :param data: PyG data object
        :param node_ratio: "auto" for automatic calculation or a numeric value to manually set node selection ratio
        :param edge_ratio: Ensures sufficient edges in the subgraph, required only if node_ratio is 'auto'
        :param strategy: Node selection strategy (e.g., "random", "high_degree", "top_pagerank", "manual", "high_betweenness")
        :param manual_nodes: Comma-separated list of node indices if using 'manual' strategy
        """
        self.data = data
        self.node_ratio = node_ratio
        self.edge_ratio = edge_ratio
        self.strategy = strategy
        self.manual_nodes = manual_nodes  # 手動選擇的節點（字串）
    
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

    
    def select_nodes(self):
        """
        Selects nodes based on the chosen strategy.
        """
        train_nodes = self.data.train_mask.nonzero().view(-1).tolist()
        node_ratio = self._calculate_node_ratio()
        num_selected = max(1, int(self.data.x.shape[0] * node_ratio))

        if self.strategy == "random":
            random.seed(42)
            return random.sample(train_nodes, num_selected)

        elif self.strategy == "high_degree":
            return self._select_high_degree_nodes(train_nodes, num_selected)

        elif self.strategy == "top_pagerank":
            return self._select_top_pagerank_nodes(train_nodes, num_selected)

        elif self.strategy == "manual":
            return self._select_manual_nodes()

        elif self.strategy == "high_betweenness":
            return self._select_high_betweeness(train_nodes, num_selected)

        else:
            raise ValueError(f"Unsupported choose_nodes strategy: {self.strategy}")


    def _select_high_degree_nodes(self, train_nodes, num_selected):
        """
        Select nodes with the highest degree centrality.
        """
        G = to_networkx(self.data, to_undirected=True)
        G = G.subgraph(train_nodes)  

        centrality = nx.degree_centrality(G)
        sorted_nodes = sorted(train_nodes, key=lambda n: centrality.get(n, 0), reverse=True)
        return sorted_nodes[:num_selected]


    def _select_top_pagerank_nodes(self, train_nodes, num_selected):
        """
        Select nodes with the highest PageRank scores.
        """
        G = to_networkx(self.data, to_undirected=True)
        G = G.subgraph(train_nodes) 

        pagerank = nx.pagerank(G)
        sorted_nodes = sorted(train_nodes, key=lambda n: pagerank.get(n, 0), reverse=True)
        return sorted_nodes[:num_selected]

    def _select_manual_nodes(self):
        """
        Select nodes manually.
        """
        if not self.manual_nodes:
            raise ValueError("Manual node selection requires a list of node indices.")

        try:
            return [int(n) for n in self.manual_nodes.split(",")]
        except ValueError:
            raise ValueError("Invalid node index format in manual selection. Use comma-separated integers.")

    def _select_high_betweeness(self, train_nodes, num_selected):
        """
        Select nodes with the highest betweenness centrality.
        """
        G = to_networkx(self.data, to_undirected=True)
        G = G.subgraph(train_nodes)  # 限制只在訓練節點上計算 centrality

        betweenness = nx.betweenness_centrality(G)
        sorted_nodes = sorted(train_nodes, key=lambda n: betweenness.get(n, 0), reverse=True)
        return sorted_nodes[:num_selected]

