import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from layers.gin_layer import GINLayer, ApplyNodeFunc, MLP
from gcn_lib.sparse import MultiSeq, PlainDynBlock, ResDynBlock, DenseDynBlock, DilatedKnnGraph
from gcn_lib.sparse import MLP as MLPpyg
from gcn_lib.sparse import GraphConv as GraphConvNet
# import torch_geometric as tg
from torch_geometric.nn import GINConv

class GINNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp_GIN']               # GIN
        learn_eps = net_params['learn_eps_GIN']              # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN'] # GIN
        readout = net_params['readout']                      # this is graph_pooling_type    
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        self.embedding_h = nn.Embedding(in_dim, hidden_dim)
        
        for layer in range(self.n_layers):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, batch_norm, residual, 0, learn_eps))

        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(self.n_layers+1):
            self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))
        
        
    def forward(self, g, h, e):
        
        h = self.embedding_h(h)
        
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h)
            hidden_rep.append(h)

        score_over_layer = 0

        for i, h in enumerate(hidden_rep):
            score_over_layer += self.linears_prediction[i](h)

        return score_over_layer
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss


class GINNet_pyg(nn.Module):

    def __init__(self, net_params):
        super().__init__()
        in_dim_node = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp_GIN']  # GIN
        learn_eps = net_params['learn_eps_GIN']  # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN']  # GIN
        readout = net_params['readout']  # this is graph_pooling_type
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.normlayers = torch.nn.ModuleList()

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)       # node feat is an integer
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        for layer in range(self.n_layers):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(GINConv(ApplyNodeFunc(mlp), 0, learn_eps))
            # note that neighbor_aggr_type can not work because the father
            # self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
            #                                dropout, batch_norm, residual, 0, learn_eps))
            if batch_norm:
                self.normlayers.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(self.n_layers + 1):
            self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))

    def forward(self, h, edge_index, e):

        h = self.embedding_h(h)

        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.n_layers):
            h_in = h
            h = self.ginlayers[i](h, edge_index)
            if self.batch_norm:
                h = self.normlayers[i](h)  # batch normalization
            h = F.relu(h)  # non-linear activation
            if self.residual:
                h = h_in + h  # residual connection
            h = F.dropout(h, self.dropout, training=self.training)
            hidden_rep.append(h)

        score_over_layer = 0

        for i, h in enumerate(hidden_rep):
            score_over_layer += self.linears_prediction[i](h)

        return score_over_layer

    def loss(self, pred, label):

        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
    def loss_proteins(self, pred, label):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(pred, label.to(torch.float))
        return loss