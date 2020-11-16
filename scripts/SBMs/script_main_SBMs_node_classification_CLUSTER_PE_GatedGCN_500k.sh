#!/bin/bash


############
# Usage
############

# bash script_main_SBMs_node_classification_CLUSTER_PE_GatedGCN_500k.sh



############
# GNNs
############

#MLP
#GCN
#GraphSage
#GatedGCN
#GAT
#MoNet
#GIN
#3WLGNN
#RingGNN



############
# SBM_CLUSTER - 2 RUNS
############

seed0=41
seed1=95
seed2=12
seed3=35
code=main_SBMs_node_classification.py 
dataset=SBM_CLUSTER
tmux new -s benchmark_resglusterpe -d
tmux send-keys "source activate pytorch1.5.0" C-m
tmux send-keys "
python $code --dataset SBM_CLUSTER --gpu_id 0 --seed 41 --config configs/SBMs_node_clustering_ResGatedGCN_pyg_PE_500k.json &
python $code --dataset SBM_CLUSTER --gpu_id 1 --seed 95 --config configs/SBMs_node_clustering_ResGatedGCN_pyg_PE_500k.json &
wait" C-m
tmux send-keys "
python $code --dataset SBM_CLUSTER --gpu_id 0 --seed 12 --config configs/SBMs_node_clustering_ResGatedGCN_pyg_PE_500k.json &
python $code --dataset SBM_CLUSTER --gpu_id 1 --seed 35 --config configs/SBMs_node_clustering_ResGatedGCN_pyg_PE_500k.json &
wait" C-m
#tmux send-keys "tmux kill-session -t benchmark" C-m









