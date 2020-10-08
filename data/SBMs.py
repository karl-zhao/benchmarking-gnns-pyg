
import time
import os
import pickle
import numpy as np
import os.path as osp
import dgl
import torch

from scipy import sparse as sp
import numpy as np
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from scipy.sparse import csr_matrix


class load_SBMsDataSetDGL(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 name,
                 split):

        self.split = split
        self.is_test = split.lower() in ['test', 'val'] 
        with open(os.path.join(data_dir, name + '_%s.pkl' % self.split), 'rb') as f:
            self.dataset = pickle.load(f)
        self.node_labels = []
        self.graph_lists = []
        self.n_samples = len(self.dataset)
        self._prepare()
    

    def _prepare(self):

        print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))

        for data in self.dataset:

            node_features = data.node_feat
            edge_list = (data.W != 0).nonzero()  # converting adj matrix to edge_list

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(node_features.size(0))
            g.ndata['feat'] = node_features.long()
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())

            # adding edge features for Residual Gated ConvNet
            #edge_feat_dim = g.ndata['feat'].size(1) # dim same as node feature dim
            edge_feat_dim = 1 # dim same as node feature dim
            g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim)

            self.graph_lists.append(g)
            self.node_labels.append(data.node_label)


    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.node_labels[idx]


class PygNodeSBMsDataset(InMemoryDataset):

    def __init__(self,
                 data_dir,
                 name,
                 split,
                 transform = None,
                 pre_transform = None,
                 meta_dict = None
                 ):

        self.split = split
        self.root = data_dir
        self.name = name
        self.is_test = split.lower() in ['test', 'val']

        self.node_labels = []
        self.graph_lists = []
        super(PygNodeSBMsDataset, self).__init__(self.root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_dir(self):
        return osp.join(self.root)

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        return [os.path.join(self.name + '_%s.pkl' % self.split)]

    @property
    def processed_file_names(self):
        return 'geometric_data_processed' + self.name + self.split + '.pt'

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self):
        with open(os.path.join(self.root, self.name + '_%s.pkl' % self.split), 'rb') as f:
            self.dataset = pickle.load(f)
        # self.n_samples = len(self.dataset)
        print("preparing graphs for the %s set..." % (self.split.upper()))
        print('Converting graphs into PyG objects...')
        pyg_graph_list = []
        for data in tqdm(self.dataset):
            node_features = data.node_feat
            edge_list = (data.W != 0).nonzero()  # converting adj matrix to edge_list
            g = Data()
            g.__num_nodes__ = node_features.size(0)
            g.edge_index = edge_list.T
            #g.edge_index = torch.from_numpy(edge_list)
            g.x = node_features.long()
            # adding edge features for Residual Gated ConvNet
            edge_feat_dim = 1
            g.edge_attr = torch.ones(g.num_edges, edge_feat_dim)
            g.y = data.node_label.to(torch.float32)
            pyg_graph_list.append(g)
        del self.dataset
        data, slices = self.collate(pyg_graph_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class SBMsDatasetpyg(InMemoryDataset):

    def __init__(self, name):
        """
            TODO
        """
        start = time.time()
        print("[I] Loading data ...")
        self.name = name
        data_dir = 'data/SBMs'
        self.train = PygNodeSBMsDataset(data_dir, name, split='train')
        self.test = PygNodeSBMsDataset(data_dir, name, split='test')
        self.val = PygNodeSBMsDataset(data_dir, name, split='val')
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    def _add_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        # self.train.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        # iter(self.train)
        self.train.graph_lists = [positional_encoding(g, pos_enc_dim, framework = 'pyg') for _, g in enumerate(dataset.train)]

        self.val.graph_lists = [positional_encoding(g, pos_enc_dim, framework = 'pyg') for _, g in enumerate(dataset.train)]
        self.test.graph_lists = [positional_encoding(g, pos_enc_dim, framework = 'pyg') for _, g in enumerate(dataset.train)]


class SBMsDatasetDGL(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            TODO
        """
        start = time.time()
        print("[I] Loading data ...")
        self.name = name
        data_dir = 'data/SBMs'
        self.train = load_SBMsDataSetDGL(data_dir, name, split='train')
        self.test = load_SBMsDataSetDGL(data_dir, name, split='test')
        self.val = load_SBMsDataSetDGL(data_dir, name, split='val')
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))




def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in SBMsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g



def positional_encoding(g, pos_enc_dim, framework = 'dgl'):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian,for the pyg
    if framework == 'pyg':
        L = get_laplacian(g.edge_index,normalization='sym',dtype = torch.float64)
        L = csr_matrix((L[1], (L[0][0], a1[0][1])), shape=(g.num_nodes, g.num_nodes))
        # Eigenvectors with scipy
        # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
        EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
        EigVec = EigVec[:, EigVal.argsort()]  # increasing order
        g.pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1].astype(np.float32)).float()
        # add astype to discards the imaginary part to satisfy the version change pytorch1.5.0
    elif framework == 'dgl':
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N
        # Eigenvectors with scipy
        # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
        EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
        EigVec = EigVec[:, EigVal.argsort()]  # increasing order
        g.ndata['pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1].astype(np.float32)).float()
        # add astype to discards the imaginary part to satisfy the version change pytorch1.5.0

    # # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L.toarray())
    # idx = EigVal.argsort() # increasing order
    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()
    return g


    
class SBMsDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/SBMs/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = torch.cat(tab_snorm_n).sqrt()  
        #tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        #tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        #snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels
    
    # prepare dense tensors for GNNs which use; such as RingGNN and 3WLGNN
    def collate_dense_gnn(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = tab_snorm_n[0][0].sqrt()  
        
        #batched_graph = dgl.batch(graphs)
    
        g = graphs[0]
        adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())        
        """
            Adapted from https://github.com/leichen2018/Ring-GNN/
            Assigning node and edge feats::
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
            The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
            The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
        """

        zero_adj = torch.zeros_like(adj)
        
        if self.name == 'SBM_CLUSTER': 
            self.num_node_type = 7
        elif self.name == 'SBM_PATTERN':
            self.num_node_type = 3
        
        # use node feats to prepare adj
        adj_node_feat = torch.stack([zero_adj for j in range(self.num_node_type)])
        adj_node_feat = torch.cat([adj.unsqueeze(0), adj_node_feat], dim=0)

        for node, node_label in enumerate(g.ndata['feat']):
            adj_node_feat[node_label.item()+1][node][node] = 1

        x_node_feat = adj_node_feat.unsqueeze(0)
        
        return x_node_feat, labels
    
    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim = 0)#.squeeze()
        deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))
    
    
    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]


    def _add_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [positional_encoding(g, pos_enc_dim, framework = 'dgl') for g in self.train.graph_lists]
        self.val.graph_lists = [positional_encoding(g, pos_enc_dim, framework = 'dgl') for g in self.val.graph_lists]
        self.test.graph_lists = [positional_encoding(g, pos_enc_dim, framework = 'dgl') for g in self.test.graph_lists]




