"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.COLLAB_edge_classification.gated_gcn_net import GatedGCNNet
from nets.COLLAB_edge_classification.gcn_net import GCNNet
from nets.COLLAB_edge_classification.gat_net import GATNet
from nets.COLLAB_edge_classification.graphsage_net import GraphSageNet
from nets.COLLAB_edge_classification.gin_net import GINNet
from nets.COLLAB_edge_classification.mo_net import MoNet as MoNet_
from nets.COLLAB_edge_classification.mlp_net import MLPNet
from nets.COLLAB_edge_classification.matrix_factorization import *


def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def MF(net_params):
    return MatrixFactorization(net_params)

def SAGE(net_params):
    return SAGENet(net_params)

def pyg_GCN(net_params):
    return pyg_GCNNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    if net_params['framework'] == 'pyg':
        MODEL_NAME = 'pyg_' + MODEL_NAME
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet,
        'MLP': MLP,
        'MF': MF,
        'pyg_SAGE': SAGE,
        'pyg_GCN': pyg_GCN,
    }
        
    return models[MODEL_NAME](net_params)