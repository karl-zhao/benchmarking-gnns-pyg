import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import dgl
import numpy as np

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""



class GCNNet_pyg(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim']  # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
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
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)       # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.dropout = dropout
        self.layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim, improved = False)
                                     for _ in range(self.n_layers)])
        if self.batch_norm:
            self.normlayers = nn.ModuleList([nn.BatchNorm1d(hidden_dim)
                                     for _ in range(self.n_layers)])
        # self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
        #                                       self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        # self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual))
        # self.MLP_layer = MLPReadout(out_dim, n_classes)
        self.MLP_layer = nn.Linear(hidden_dim, n_classes, bias=True)

    def forward(self, h, edge_index, e, h_pos_enc=None):
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        # GCN
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
            h = h + h_pos_enc
        for i in range(self.n_layers):
            h_in = h
            h = self.layers[i](h, edge_index)
            if self.batch_norm:
                h = self.normlayers[i](h)  # batch normalization
            h = F.relu(h)  # non-linear activation
            if self.residual:
                h = h_in + h  # residual connection
            h = F.dropout(h, self.dropout, training=self.training)
        # i = 0
        # for conv in self.layers:
        #     h_in = h
        #     h = conv(h, e)
        #     if self.batch_norm:
        #         h = self.normlayers[i](h)  # batch normalization
        #         i += 1
        #     h = F.relu(h)
        #     if self.residual:
        #         h = h_in + h  # residual connection
        #     h = F.dropout(h, self.dropout, training=self.training)
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











