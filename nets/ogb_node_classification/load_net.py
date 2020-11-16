"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.ogb_node_classification.gated_gcn_net import GatedGCNNet, GatedGCNNet_pyg, ResGatedGCNNet_pyg
from nets.ogb_node_classification.gcn_net import GCNNet_pyg
from nets.ogb_node_classification.gat_net import GATNet_pyg
from nets.ogb_node_classification.graphsage_net import GraphSageNet, GraphSageNet_pyg
from nets.ogb_node_classification.mlp_net import MLPNet, MLPNet_pyg
from nets.ogb_node_classification.gin_net import GINNet, GINNet_pyg
from nets.ogb_node_classification.mo_net import MoNet as MoNet_, MoNetNet_pyg

from gcn_lib.sparse import MultiSeq, PlainDynBlock, ResDynBlock, DenseDynBlock, DilatedKnnGraph
from gcn_lib.sparse import MLP as MLPpyg
from gcn_lib.sparse import GraphConv as GraphConvNet


def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)


def GIN_pyg(net_params):
    model = GINNet_pyg(net_params)
    if net_params['neighbor_aggr_GIN'] == 'mean':
        model.aggr = str('mean')
    elif net_params['neighbor_aggr_GIN'] == 'max':
        model.aggr = str('max')
    return model

def MLP_pyg(net_params):
    return MLPNet_pyg(net_params)

def GCN_pyg(net_params):
    return GCNNet_pyg(net_params)

def GatedGCN_pyg(net_params):
    return GatedGCNNet_pyg(net_params)


def ResGatedGCN_pyg(net_params):
    return ResGatedGCNNet_pyg(net_params)

def GAT_pyg(net_params):
    return GATNet_pyg(net_params)
# self.head = GraphConv(opt.in_channels, channels, conv, act, norm, bias, heads)
def GraphSage_pyg(net_params):
    return GraphSageNet_pyg(net_params)

def MoNet_pyg(net_params):
    return MoNetNet_pyg(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'MLP': MLP,
        'GIN': GIN,
        'MoNet': MoNet,
        'MLP_pyg': MLP_pyg,
        'GIN_pyg': GIN_pyg,
        'GCN_pyg': GCN_pyg,
        'GatedGCN_pyg': GatedGCN_pyg,
        'GAT_pyg': GAT_pyg,
        'GraphSage_pyg': GraphSage_pyg,
        'MoNet_pyg': MoNet_pyg,
        'ResGatedGCN_pyg': ResGatedGCN_pyg
    }
        
    return models[MODEL_NAME](net_params)