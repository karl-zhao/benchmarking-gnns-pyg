import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer, ResGatedGCNLayer
from layers.mlp_readout_layer import MLPReadout
from torch_geometric.nn import GatedGraphConv

class GatedGCNNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        in_dim_edge = 1 # edge_dim (feat is a float)
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim) # edge feat is a float
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                    self.batch_norm, self.residual) for _ in range(n_layers) ])
        self.MLP_layer = MLPReadout(hidden_dim, n_classes)


    def forward(self, g, h, e, h_pos_enc=None):

        # input embedding
        h = self.embedding_h(h)
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float()) 
            h = h + h_pos_enc
        e = self.embedding_e(e)
        
        # res gated convnets
        for conv in self.layers:
            h, e = conv(g, h, e)

        # output
        h_out = self.MLP_layer(h)

        return h_out
        

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss

class ResGatedGCNNet_pyg(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim']  # node_dim (feat is an integer)
        in_dim_edge = 1  # edge_dim (feat is a float)
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        self.dropout = net_params['dropout']
        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)  # node feat is an integer
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)  # edge feat is a float
        self.layers = nn.ModuleList([ResGatedGCNLayer(hidden_dim, hidden_dim, self.dropout,
                                                   self.batch_norm, self.residual) for _ in range(self.n_layers)])
        # self.layers = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
        #                                            self.batch_norm, self.residual) for _ in range(n_layers)])
        if self.batch_norm:
            self.normlayers_h = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(self.n_layers)])
            self.normlayers_e = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(self.n_layers)])
        self.MLP_layer = MLPReadout(hidden_dim, n_classes)

    def forward(self, h, edge_index, e, h_pos_enc=None):

        # input embedding
        h = self.embedding_h(h)
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
            h = h + h_pos_enc
        e = self.embedding_e(e)
        # res gated convnets
        for i in range(self.n_layers):
            h_in = h
            e_in = e
            h, e = self.layers[i](h, edge_index, e)
            if self.batch_norm:
                h = self.normlayers_h[i](h)
                e = self.normlayers_e[i](e)  # batch normalization
            if self.residual:
                h = h_in + h  # residual connection
                e = e_in + e
            h = F.dropout(h, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)
        # output
        h_out = self.MLP_layer(h)

        return h_out

    # if self.batch_norm:
    #     h = self.bn_node_h(h)  # batch normalization
    #     e = self.bn_node_e(e)  # batch normalization
    #
    # h = F.relu(h)  # non-linear activation
    # e = F.relu(e)  # non-linear activation
    #
    # if self.residual:
    #     h = h_in + h  # residual connection
    #     e = e_in + e  # residual connection
    #
    # h = F.dropout(h, self.dropout, training=self.training)
    # e = F.dropout(e, self.dropout, training=self.training)

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss





"""
    Gated Graph Sequence Neural Networks
    An Experimental Study of Neural Networks for Variable Graphs 
    Li Y, Tarlow D, Brockschmidt M, et al. Gated graph sequence neural networks[J]. arXiv preprint arXiv:1511.05493, 2015.
    https://arxiv.org/abs/1511.05493
    Note that the pyg and dgl of the gatedGCN are different models.
"""
class GatedGCNNet_pyg(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim']  # node_dim (feat is an integer)
        in_dim_edge = 1  # edge_dim (feat is a float)
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        self.dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)  # node feat is an integer
        # self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)  # edge feat is a float
        self.layers = nn.ModuleList([GatedGraphConv(hidden_dim, n_layers, aggr = 'add')])
        # self.layers = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
        #                                            self.batch_norm, self.residual) for _ in range(n_layers)])
        if self.batch_norm:
            self.normlayers = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
        self.MLP_layer = MLPReadout(hidden_dim, n_classes)

    def forward(self, h, edge_index, e, h_pos_enc=None):

        # input embedding
        h = self.embedding_h(h)
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
            h = h + h_pos_enc
        # e = self.embedding_e(e)
        # res gated convnets
        for conv in self.layers:
            h_in = h
            h = conv(h, edge_index, e)
        if self.batch_norm:
            h = self.normlayers[0](h)
        if self.residual:
            h = h_in + h  # residual connection
        h = F.dropout(h, self.dropout, training=self.training)
        # output
        h_out = self.MLP_layer(h)

        return h_out

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss