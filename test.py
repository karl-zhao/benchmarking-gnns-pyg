# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl

from scipy import sparse as sp
import numpy as np


# # coding=gbk
# from tqdm import trange
# from random import random,randint
# import time
#
# with trange(100) as t:
#     for i in t:
#         #t.set_description("GEN111 %i" % i)
#         t.set_postfix(loss=8,gen=randint(1,999),str="h",lst=[1,2],lst11=[1,2],loss11=8)
#         time.sleep(0.1)
# t.close()

# import dgl
# import torch as th
# # 4 nodes, 3 edges
# # g1 = dgl.graph((th.tensor([0, 1, 2]), th.tensor([1, 2, 3])))
# def positional_encoding(g, pos_enc_dim):
#     """
#         Graph positional encoding v/ Laplacian eigenvectors
#     """
#
#     # Laplacian
#     A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
#     N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
#     L = sp.eye(g.number_of_nodes()) - N * A * N
#
#     # Eigenvectors with numpy
#     EigVal, EigVec = np.linalg.eig(L.toarray())
#     idx = EigVal.argsort()  # increasing order from min to max order index
#     EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
#     g.ndata['pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
#
#     # # Eigenvectors with scipy
#     # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
#     # EigVec = EigVec[:, EigVal.argsort()] # increasing order
#     # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()
#
#     return g
#
#
# def message_func(edges):
#     Bh_j = edges.src['Bh']
#     e_ij = edges.data['Ce'] + edges.src['Dh'] + edges.dst['Eh']  # e_ij = Ce_ij + Dhi + Ehj
#     edges.data['e'] = e_ij
#     return {'Bh_j': Bh_j, 'e_ij': e_ij}
#
#
# def reduce_func(nodes):
#     Ah_i = nodes.data['Ah']#这个对只有出去，没有进来的点没有。这个时候只能是0，还不如加个自循环，用messange来做
#     Bh_j = nodes.mailbox['Bh_j']
#     e = nodes.mailbox['e_ij']
#     sigma_ij = torch.sigmoid(e.float())  # sigma_ij = sigmoid(e_ij)
#     # h = Ah_i + torch.mean( sigma_ij * Bh_j, dim=1 ) # hi = Ahi + mean_j alpha_ij * Bhj
#     h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1) / (torch.sum(sigma_ij,
#                                                               dim=1) + 1e-6)  # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention
#     return {'h': h}
# g1 = dgl.DGLGraph()
# g1.add_nodes(4)
# g1.add_edges([0, 1, 2], [1, 2, 3])
# # g1.ndata['h'] = th.randn((4, 3),dtype=th.float32)
# # g1.ndata['Ah'] = th.randn((4, 3),dtype=th.float32)
# # g1.ndata['Bh'] = th.randn((4, 3),dtype=th.float32)
# # g1.ndata['Dh'] = th.randn((4, 3),dtype=th.float32)
# # g1.ndata['Eh'] = th.randn((4, 3),dtype=th.float32)
# # g1.edata['e']=th.randn((3, 3),dtype=th.float32)
# # g1.edata['Ce'] = th.randn((3, 3),dtype=th.float32)
# g1.ndata['h'] = th.reshape(th.arange(1, 13), (4, 3))
# g1.ndata['Ah'] = th.reshape(th.arange(13, 25), (4, 3))
# g1.ndata['Bh'] = th.reshape(th.arange(26, 38), (4, 3))
# g1.ndata['Dh'] = th.reshape(th.arange(39, 51), (4, 3))
# g1.ndata['Eh'] = th.reshape(th.arange(52, 64), (4, 3))
# g1.edata['e'] = th.reshape(th.arange(65, 74), (3, 3))
# g1.edata['Ce'] = th.reshape(th.arange(75, 84), (3, 3))
# positional_encoding(g1, 3)
# # 3 nodes, 4 edges
# g2 = dgl.DGLGraph()
# g2.add_nodes(3)
# g2.add_edges([0, 0, 0, 1], [0, 1, 2, 0])
# g2.ndata['h'] = th.reshape(th.arange(101, 110), (3, 3))
# g2.ndata['Ah'] = th.reshape(th.arange(113, 122), (3, 3))
# g2.ndata['Bh'] = th.reshape(th.arange(126, 135), (3, 3))
# g2.ndata['Dh'] = th.reshape(th.arange(139, 148), (3, 3))
# g2.ndata['Eh'] = th.reshape(th.arange(152, 161), (3, 3))
# g2.edata['e'] = th.reshape(th.arange(165, 177), (4, 3))
# g2.edata['Ce'] = th.reshape(th.arange(175, 187), (4, 3))
# bg = dgl.batch([g1, g2])
# bg.update_all(message_func, reduce_func)
# bg.ndata['h']
# a = 1+1
# # g3 = dgl.graph((th.tensor([0, 1, 2]), th.tensor([1, 2, 3])))
# # g4 = dgl.graph((th.tensor([0, 0, 0, 1]), th.tensor([0, 1, 2, 0])))
# # bg = dgl.batch([g3, g4], edge_attrs=None)
#
# # import dgl
# # import torch as th
# # g1 = dgl.DGLGraph()
# # g1.add_nodes(2)                                # Add 2 nodes
# # g1.add_edge(0, 1)                              # Add edge 0 -> 1
# # g1.ndata['hv'] = th.tensor([[0.], [1.]])       # Initialize node features
# # g1.edata['he'] = th.tensor([[0.]])             # Initialize edge features
# # g2 = dgl.DGLGraph()
# # g2.add_nodes(3)                                # Add 3 nodes
# # g2.add_edges([0, 2], [1, 1])                   # Add edges 0 -> 1, 2 -> 1
# # g2.ndata['hv'] = th.tensor([[2.], [3.], [4.]]) # Initialize node features
# # g2.edata['he'] = th.tensor([[1.], [2.]])       # Initialize edge features
# # bg = dgl.batch([g1, g2], edge_attrs=None)
# import time
# try:
#     while True:
#         print("你好")
#         time.sleep(1)
# except KeyboardInterrupt:
#         print('aa')
#
# print("好!")
# import numpy as np
# import torch
# import pickle
# import time
# import os
# import matplotlib.pyplot as plt
# if not os.path.isfile('molecules.zip'):
#     print('downloading..')
#     !curl https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1 -o molecules.zip -J -L -k
#     !unzip molecules.zip -d ../
#     # !tar -xvf molecules.zip -C ../
# else:
#     print('File already downloaded')
from tqdm import tqdm
import time
for i in tqdm(range(10000)):
    time.sleep(0.001)

