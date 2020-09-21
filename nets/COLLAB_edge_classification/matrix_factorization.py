import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from torch_geometric.nn import GCNConv, SAGEConv

from layers.mlp_readout_layer import MLPReadout

class pyg_GCNNet(torch.nn.Module):
    def __init__(self, net_params):
        super(pyg_GCNNet, self).__init__()
        in_channels = net_params['in_dim']
        hidden_channels = net_params['hidden_dim']
        out_channels = net_params['out_dim']
        #in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        num_layers = net_params['L']
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, 1))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

    def edge_predictor(self, h_i, h_j):
        x = h_i * h_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def loss(self, pos_out, neg_out):
        pos_loss = -torch.log(pos_out + 1e-15).mean()  # positive samples
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()  # negative samples
        loss = pos_loss + neg_loss

        return loss



class SAGENet(torch.nn.Module):
    def __init__(self, net_params):
        super(SAGENet, self).__init__()
        in_channels = net_params['in_dim']
        hidden_channels = net_params['hidden_dim']
        out_channels = net_params['out_dim']
        # in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        num_layers = net_params['L']
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, 1))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

    def edge_predictor(self, h_i, h_j):
        x = h_i * h_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def loss(self, pos_out, neg_out):
        pos_loss = -torch.log(pos_out + 1e-15).mean()  # positive samples
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()  # negative samples
        loss = pos_loss + neg_loss

        return loss


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class MatrixFactorization(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_embs = net_params['num_embs']
        hidden_dim = net_params['hidden_dim']
        self.device = net_params['device']
        
        # MF trains a hidden embedding per graph node
        self.emb = torch.nn.Embedding(num_embs, hidden_dim)
        
        self.readout_mlp = MLPReadout(2*hidden_dim, 1)

    def forward(self, g, h, e):
        # Return the entire node embedding matrix
        return self.emb.weight
        
    def edge_predictor(self, h_i, h_j):
        x = torch.cat([h_i, h_j], dim=1)
        x = self.readout_mlp(x)
        
        return torch.sigmoid(x)
    
    def loss(self, pos_out, neg_out):
        pos_loss = -torch.log(pos_out + 1e-15).mean()  # positive samples
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()  # negative samples
        loss = pos_loss + neg_loss
        
        return loss
