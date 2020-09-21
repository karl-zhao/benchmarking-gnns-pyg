#!/usr/bin/env python
# coding: utf-8

# # Notebook for preparing and saving MOLECULAR graphs

# In[1]:


import numpy as np
import torch
import pickle
import time
import os
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


print(torch.__version__)


# # Download ZINC dataset

# In[1]:


#!unzip molecules.zip -d ../


# In[3]:


if not os.path.isfile('molecules.zip'):
    print('downloading..')
    os.system('curl https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1 -o molecules.zip -J -L -k')
    os.system('unzip molecules.zip -d ../')
    # !tar -xvf molecules.zip -C ../
else:
    print('File already downloaded')
    


# # Convert to DGL format and save with pickle

# In[4]:


import os
os.chdir('../../') # go to root folder of the project
print(os.getcwd())


# In[5]:


import pickle

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

from data.molecules import MoleculeDatasetDGL ,MoleculeDatasetpyg

from data.data import LoadData
from torch.utils.data import DataLoader
from data.molecules import MoleculeDataset


# In[6]:
framwork = 'pyg'

DATASET_NAME = 'ZINC'
dataset = MoleculeDatasetDGL(DATASET_NAME) if 'dgl' == framwork else MoleculeDatasetpyg(DATASET_NAME)

# In[7]:


def plot_histo_graphs(dataset, title):
    # histogram of graph sizes
    graph_sizes = []
    for graph in dataset:
        graph_sizes.append(graph.num_nodes) if framwork == 'pyg' else graph_sizes.append(graph[0].number_of_nodes())
    plt.figure(1)
    plt.hist(graph_sizes, bins=20)
    plt.title(title)
    plt.show()
    graph_sizes = torch.Tensor(graph_sizes)
    print('min/max :',graph_sizes.min().long().item(),graph_sizes.max().long().item())
    
#plot_histo_graphs(dataset.train,'trainset')
plot_histo_graphs(dataset.val,'valset')
plot_histo_graphs(dataset.test,'testset')


# In[8]:


#print(len(dataset.train))
print(len(dataset.val))
print(len(dataset.test))

#print(dataset.train[0])
print(dataset.val[0])
print(dataset.test[0])


# In[9]:


num_atom_type = 28
num_bond_type = 4


# In[10]:


# start = time.time()
#
# with open('data/molecules/ZINC_dgl.pkl','wb') as f:
#         pickle.dump([dataset.train,dataset.val,dataset.test,num_atom_type,num_bond_type],f)
# print('Time (sec):',time.time() - start)



# # Test load function

# In[11]:


# DATASET_NAME = 'ZINC'
# dataset = LoadData(DATASET_NAME, framwork)
# trainset, valset, testset = dataset.train, dataset.val, dataset.test


# In[12]:
from torch_geometric.data import DataLoader

loader = DataLoader(dataset.val, batch_size=32, shuffle=True)
for batch in loader:
    print(batch)
    print(batch.y)
    print(len(batch.y))

# batch_size = 10
# collate = MoleculeDataset.collate
# print(MoleculeDataset)
# train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate)


# In[ ]:





# In[ ]:




