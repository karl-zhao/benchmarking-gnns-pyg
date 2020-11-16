"""
    File to load dataset based on user control from main file
"""
from data.molecules import *
from data.SBMs import SBMsDataset, SBMsDatasetpyg

from data.planetoids import PlanetoidDataset
from data.ogbn import ogbnDatasetpyg


def LoadData(DATASET_NAME, use_node_embedding = False, framework = 'dgl'):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    # handling for MNIST or CIFAR Superpixels
    
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC':
        return MoleculeDataset(DATASET_NAME) if 'dgl' == framework else MoleculeDatasetpyg(DATASET_NAME)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS: 
        return SBMsDatasetpyg(DATASET_NAME) if 'pyg' == framework else SBMsDataset(DATASET_NAME)

    if DATASET_NAME in ['Cora', 'Citeseer', 'Pubmed']:
        return PlanetoidDataset(DATASET_NAME, use_node_embedding = use_node_embedding)

    if DATASET_NAME in ['ogbn-arxiv', 'ogbn-proteins', 'ogbn-mag', 'ogbn-products']:
        return ogbnDatasetpyg(name=DATASET_NAME, use_node_embedding = use_node_embedding)


    