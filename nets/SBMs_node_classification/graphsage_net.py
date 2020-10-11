import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.graphsage_layer import GraphSageLayer
from layers.mlp_readout_layer import MLPReadout
from torch_geometric.nn import SAGEConv

class GraphSageNet(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']   
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.readout = net_params['readout']
        self.n_classes = n_classes
        self.device = net_params['device']
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, batch_norm, residual) for _ in range(n_layers-1)])
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        

    def forward(self, g, h, e):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        # graphsage
        for conv in self.layers:
            h = conv(g, h)

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

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
    # the implementation of dgl and pyg are different
"""
class GraphSageNet_pyg(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim']  # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        self.n_layers = net_params['L']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.readout = net_params['readout']
        self.n_classes = n_classes
        self.device = net_params['device']

        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)  # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.dropout = nn.Dropout(p=dropout)
        if self.batch_norm:
            self.batchnorm_h = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(self.n_layers - 1)])
            self.batchnorm_h.append(nn.BatchNorm1d(out_dim))
        self.layers = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in
                                     range(self.n_layers - 1)])
        self.layers.append(
            SAGEConv(hidden_dim, out_dim))
        # self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
        #                                             dropout, aggregator_type, batch_norm, residual) for _ in
        #                              range(n_layers - 1)])
        # self.layers.append(
        #     GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        if aggregator_type == 'maxpool':
            self.aggr = 'max'
        elif aggregator_type == 'mean':
            self.aggr = 'mean'
        # to do

    def forward(self, h, edge_index, e):
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        # graphsage
        for i in range(self.n_layers):
            h_in = h
            h = self.dropout(h)
            h = self.layers[i](h, edge_index)
            if self.batch_norm:
                h = self.batchnorm_h[i](h)
            #h = F.relu(h)
            if self.residual:
                h = h_in + h  # residual connection

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
