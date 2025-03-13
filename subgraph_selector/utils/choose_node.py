import random
import networkx as nx
from torch_geometric.utils import to_networkx

class ChooseNodeSelector:
    """
    A class to select nodes from a graph dataset based on different strategies.
    """

    def __init__(self, data, node_ratio=0.01, strategy="random", manual_nodes=None):
        """
        :param data: PyG data object
        :param node_ratio: Ratio of nodes to select
        :param strategy: Node selection strategy (e.g., "random", "high_degree", "top_pagerank", "manual")
        :param manual_nodes: Comma-separated list of node indices if using 'manual' strategy
        """
        self.data = data
        self.node_ratio = node_ratio
        self.strategy = strategy
        self.manual_nodes = manual_nodes  # 手動選擇的節點（字串）

    def select_nodes(self):
        """
        Selects nodes based on the chosen strategy.
        """
        train_nodes = self.data.train_mask.nonzero().view(-1).tolist()
        num_selected = max(1, int(self.data.x.shape[0] * self.node_ratio))

        if self.strategy == "random":
            return random.sample(train_nodes, num_selected)

        elif self.strategy == "high_degree":
            return self.select_high_degree_nodes(train_nodes, num_selected)

        elif self.strategy == "top_pagerank":
            return self.select_top_pagerank_nodes(train_nodes, num_selected)

        elif self.strategy == "manual":
            return self.select_manual_nodes()

        else:
            raise ValueError(f"Unsupported choose_nodes strategy: {self.strategy}")

    # 還沒看
    def select_high_degree_nodes(self, train_nodes, num_selected):
        """
        Select nodes with the highest degree centrality.
        """
        G = to_networkx(self.data, to_undirected=True)
        G = G.subgraph(train_nodes)  

        centrality = nx.degree_centrality(G)
        sorted_nodes = sorted(train_nodes, key=lambda n: centrality.get(n, 0), reverse=True)
        return sorted_nodes[:num_selected]


    def select_top_pagerank_nodes(self, train_nodes, num_selected):
        """
        Select nodes with the highest PageRank scores.
        """
        G = to_networkx(self.data, to_undirected=True)
        G = G.subgraph(train_nodes) 

        pagerank = nx.pagerank(G)
        sorted_nodes = sorted(train_nodes, key=lambda n: pagerank.get(n, 0), reverse=True)
        return sorted_nodes[:num_selected]

    def select_manual_nodes(self):
        """
        Select nodes manually.
        """
        if not self.manual_nodes:
            raise ValueError("Manual node selection requires a list of node indices.")

        try:
            return [int(n) for n in self.manual_nodes.split(",")]
        except ValueError:
            raise ValueError("Invalid node index format in manual selection. Use comma-separated integers.")
