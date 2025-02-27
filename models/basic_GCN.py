import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1=64, hidden_channels2=32, out_channels=4):
        """
        Initializes a 3-layer GCN model.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels1, cached=True, normalize=True)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2, cached=True, normalize=True)
        self.conv3 = GCNConv(hidden_channels2, out_channels, cached=True, normalize=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        return x



class GCN2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2):
        """
        Initializes a 2-layer GCN model.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True, normalize=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x



# test
if __name__ == "__main__":
    from torch_geometric.datasets import FacebookPagePage

    dataset = FacebookPagePage(root='/tmp/FacebookPagePage')
    data = dataset[0]  # Get the first graph
    in_channels = dataset.num_features  # Automatically determine input feature size

    model = GCN3(in_channels=in_channels)
    print(model)
