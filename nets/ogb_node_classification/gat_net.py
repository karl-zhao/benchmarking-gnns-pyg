import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from torch_geometric.typing import OptPairTensor

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer
from layers.mlp_readout_layer import MLPReadout
from torch_geometric.nn import GATConv



class GATNet_pyg(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim']  # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']

        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim * num_heads)
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim * num_heads)       # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList([GATConv(hidden_dim * num_heads, hidden_dim, num_heads,
                                              dropout = dropout) for _ in range(self.n_layers - 1)])
        self.layers.append(GATConv(hidden_dim * num_heads, out_dim, 1, dropout))
        if self.batch_norm:
            self.batchnorm_h = nn.ModuleList([nn.BatchNorm1d(hidden_dim * num_heads) for _ in range(self.n_layers - 1)])
            self.batchnorm_h.append(nn.BatchNorm1d(out_dim))
        # self.layers = nn.ModuleList([GATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
        #                                       dropout, self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        # self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))
        # self.MLP_layer = MLPReadout(out_dim, n_classes)
        self.MLP_layer = nn.Linear(out_dim, n_classes, bias=True)

    def forward(self, h, edge_index, e, h_pos_enc = None):
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
            h = h + h_pos_enc
        # GAT
        for i in range(self.n_layers):
            h_in = h
            h: OptPairTensor = (h, h) # make cat the value not simple add it.
            h = self.layers[i](h, edge_index)
            if self.batch_norm:
                h = self.batchnorm_h[i](h)
            # if self.activation:
            h = F.elu(h)
            if self.residual:
                h = h_in + h  # residual connection
        # output
        h_out = self.MLP_layer(h)

        return h_out

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

    def loss_proteins(self, pred, label):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(pred, label.to(torch.float))
        return loss
