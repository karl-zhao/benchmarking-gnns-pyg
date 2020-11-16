import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import dgl

import numpy as np

"""
    GMM: Gaussian Mixture Model Convolution layer
    Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs (Federico Monti et al., CVPR 2017)
    https://arxiv.org/pdf/1611.08402.pdf
"""

from layers.gmm_layer import GMMLayer
from layers.mlp_readout_layer import MLPReadout
from torch_geometric.nn import GMMConv

class MoNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.name = 'MoNet'
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        kernel = net_params['kernel']                       # for MoNet
        dim = net_params['pseudo_dim_MoNet']                # for MoNet
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']                            
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']  
        self.device = net_params['device']
        self.n_classes = n_classes
        
        aggr_type = "sum"                                    # default for MoNet
        
        self.embedding_h = nn.Embedding(in_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Hidden layer
        for _ in range(n_layers-1):
            self.layers.append(GMMLayer(hidden_dim, hidden_dim, dim, kernel, aggr_type,
                                        dropout, batch_norm, residual))
            self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
            
        # Output layer
        self.layers.append(GMMLayer(hidden_dim, out_dim, dim, kernel, aggr_type,
                                    dropout, batch_norm, residual))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e):
        h = self.embedding_h(h)
        
        # computing the 'pseudo' named tensor which depends on node degrees
        g.ndata['deg'] = g.in_degrees()
        g.apply_edges(self.compute_pseudo)
        pseudo = g.edata['pseudo'].to(self.device).float()
        
        for i in range(len(self.layers)):
            h = self.layers[i](g, h, self.pseudo_proj[i](pseudo))

        return self.MLP_layer(h)
    
    def compute_pseudo(self, edges):
        # compute pseudo edge features for MoNet
        # to avoid zero division in case in_degree is 0, we add constant '1' in all node degrees denoting self-loop
        srcs = 1/np.sqrt(edges.src['deg']+1)
        dsts = 1/np.sqrt(edges.dst['deg']+1)
        pseudo = torch.cat((srcs.unsqueeze(-1), dsts.unsqueeze(-1)), dim=1)
        return {'pseudo': pseudo}
        
    def loss(self, pred, label):

        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

"""
    GMM: Gaussian Mixture Model Convolution layer
    Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs (Federico Monti et al., CVPR 2017)
    https://arxiv.org/pdf/1611.08402.pdf
"""
class MoNetNet_pyg(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.name = 'MoNet'
        in_dim_node = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        kernel = net_params['kernel']  # for MoNet
        dim = net_params['pseudo_dim_MoNet']  # for MoNet
        n_classes = net_params['n_classes']
        self.dropout = net_params['dropout']
        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        self.n_classes = n_classes
        self.dim = dim
        # aggr_type = "sum"  # default for MoNet
        aggr_type = "mean"

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)       # node feat is an integer
        # self.embedding_e = nn.Linear(1, dim)  # edge feat is a float
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()
        self.batchnorm_h = nn.ModuleList()
        # Hidden layer
        for _ in range(self.n_layers - 1):
            self.layers.append(GMMConv(hidden_dim, hidden_dim, dim, kernel, separate_gaussians = False ,aggr = aggr_type,
                                        root_weight = True, bias = True))
            if self.batch_norm:
                self.batchnorm_h.append(nn.BatchNorm1d(hidden_dim))
            self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        # Output layer
        self.layers.append(GMMConv(hidden_dim, out_dim, dim, kernel, separate_gaussians = False ,aggr = aggr_type,
                                        root_weight = True, bias = True))
        if self.batch_norm:
            self.batchnorm_h.append(nn.BatchNorm1d(out_dim))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        self.MLP_layer = MLPReadout(out_dim, n_classes)
        # to do

    def forward(self, h, edge_index, e):
        h = self.embedding_h(h)
        edge_weight = torch.ones((edge_index.size(1),),
                                         device = edge_index.device)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=h.size(0))
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        pseudo = torch.cat((deg_inv_sqrt[row].unsqueeze(-1), deg_inv_sqrt[col].unsqueeze(-1)), dim=1)

        for i in range(self.n_layers):
            h_in = h
            h = self.layers[i](h, edge_index, self.pseudo_proj[i](pseudo))
            if self.batch_norm:
                h = self.batchnorm_h[i](h)  # batch normalization
            h = F.relu(h)  # non-linear activation
            if self.residual:
                h = h_in + h  # residual connection
            h = F.dropout(h, self.dropout, training=self.training)

        return self.MLP_layer(h)

    def loss(self, pred, label):

        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss