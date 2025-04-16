import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN2Classifier(torch.nn.Module):
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


class GCN2Regressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        """
        2-layer GCN model for node regression.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True, normalize=True)
        self.conv2 = GCNConv(hidden_channels, 1, cached=True, normalize=True)  # 回歸輸出 1 個數值

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)  # 直接輸出回歸值
        return x.view(-1)  # 展平成一維



class GCN3Classifier(torch.nn.Module):
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


class GCN3Regressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1=64, hidden_channels2=32):
        """
        3-layer GCN model for node regression.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels1, cached=True, normalize=True)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2, cached=True, normalize=True)
        self.conv3 = GCNConv(hidden_channels2, 1, cached=True, normalize=True)  # 回歸輸出 1 個數值

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)  # 直接輸出回歸值
        return x.view(-1)  # 展平成一維
