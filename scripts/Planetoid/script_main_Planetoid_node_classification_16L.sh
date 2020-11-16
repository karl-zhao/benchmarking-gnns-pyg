#!/bin/bash


############
# Usage
############

# bash script_main_Planetoid_node_classification_4L.sh



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
code=main_Planetoid_node_classification.py
tmux new -s benchmark_node_classification_p -d
tmux send-keys "source activate pytorch1.5.0" C-m
tmux send-keys "export CUDA_VISIBLE_DEVICES = 0,1" C-m
dataset=Cora11
dataset=Pubmed1
dataset=Pubmed1

tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 41 --config configs/Cora_node_classification_MLP_pyg_460k.json &
python $code --dataset Cora --gpu_id 1 --seed 95 --config configs/Cora_node_classification_MLP_pyg_460k.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 12 --config configs/Cora_node_classification_MLP_pyg_460k.json &
python $code --dataset Cora --gpu_id 1 --seed 35 --config configs/Cora_node_classification_MLP_pyg_460k.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 41 --config configs/Cora_node_classification_GCN_pyg_460k.json &
python $code --dataset Cora --gpu_id 1 --seed 95 --config configs/Cora_node_classification_GCN_pyg_460k.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 12 --config configs/Cora_node_classification_GCN_pyg_460k.json &
python $code --dataset Cora --gpu_id 1 --seed 35 --config configs/Cora_node_classification_GCN_pyg_460k.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 41 --config configs/Cora_node_classification_ResGatedGCN_pyg_460k.json &
python $code --dataset Cora --gpu_id 1 --seed 95 --config configs/Cora_node_classification_ResGatedGCN_pyg_460k.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 12 --config configs/Cora_node_classification_ResGatedGCN_pyg_460k.json &
python $code --dataset Cora --gpu_id 1 --seed 35 --config configs/Cora_node_classification_ResGatedGCN_pyg_460k.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 41 --config configs/Cora_node_classification_MoNet_pyg_460k.json &
python $code --dataset Cora --gpu_id 1 --seed 95 --config configs/Cora_node_classification_MoNet_pyg_460k.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 12 --config configs/Cora_node_classification_MoNet_pyg_460k.json &
python $code --dataset Cora --gpu_id 1 --seed 35 --config configs/Cora_node_classification_MoNet_pyg_460k.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 41 --config configs/Cora_node_classification_GIN_pyg_460.json &
python $code --dataset Cora --gpu_id 1 --seed 95 --config configs/Cora_node_classification_GIN_pyg_460.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 12 --config configs/Cora_node_classification_GIN_pyg_460.json &
python $code --dataset Cora --gpu_id 1 --seed 35 --config configs/Cora_node_classification_GIN_pyg_460.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 41 --config configs/Cora_node_classification_GatedGCN_pyg_460k.json &
python $code --dataset Cora --gpu_id 1 --seed 95 --config configs/Cora_node_classification_GatedGCN_pyg_460k.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 12 --config configs/Cora_node_classification_GatedGCN_pyg_460k.json &
python $code --dataset Cora --gpu_id 1 --seed 35 --config configs/Cora_node_classification_GatedGCN_pyg_460k.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 41 --config configs/Cora_node_classification_GraphSage_pyg_460k.json &
python $code --dataset Cora --gpu_id 1 --seed 95 --config configs/Cora_node_classification_GraphSage_pyg_460k.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 12 --config configs/Cora_node_classification_GraphSage_pyg_460k.json &
python $code --dataset Cora --gpu_id 1 --seed 35 --config configs/Cora_node_classification_GraphSage_pyg_460k.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 41 --config configs/Cora_node_classification_GAT_pyg_460k.json &
python $code --dataset Cora --gpu_id 1 --seed 95 --config configs/Cora_node_classification_GAT_pyg_460k.json &
wait" C-m
tmux send-keys "
python $code --dataset Cora --gpu_id 0 --seed 12 --config configs/Cora_node_classification_GAT_pyg_460k.json &
python $code --dataset Cora --gpu_id 1 --seed 35 --config configs/Cora_node_classification_GAT_pyg_460k.json &
wait" C-m
#'Citeseer', 'Pubmed
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 41 --config configs/Citeseer_node_classification_MLP_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 95 --config configs/Citeseer_node_classification_MLP_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 12 --config configs/Citeseer_node_classification_MLP_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 35 --config configs/Citeseer_node_classification_MLP_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 41 --config configs/Citeseer_node_classification_GCN_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 95 --config configs/Citeseer_node_classification_GCN_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 12 --config configs/Citeseer_node_classification_GCN_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 35 --config configs/Citeseer_node_classification_GCN_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 41 --config configs/Citeseer_node_classification_ResGatedGCN_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 95 --config configs/Citeseer_node_classification_ResGatedGCN_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 12 --config configs/Citeseer_node_classification_ResGatedGCN_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 35 --config configs/Citeseer_node_classification_ResGatedGCN_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 41 --config configs/Citeseer_node_classification_MoNet_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 95 --config configs/Citeseer_node_classification_MoNet_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 12 --config configs/Citeseer_node_classification_MoNet_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 35 --config configs/Citeseer_node_classification_MoNet_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 41 --config configs/Citeseer_node_classification_GIN_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 95 --config configs/Citeseer_node_classification_GIN_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 12 --config configs/Citeseer_node_classification_GIN_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 35 --config configs/Citeseer_node_classification_GIN_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 41 --config configs/Citeseer_node_classification_GatedGCN_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 95 --config configs/Citeseer_node_classification_GatedGCN_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 12 --config configs/Citeseer_node_classification_GatedGCN_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 35 --config configs/Citeseer_node_classification_GatedGCN_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 41 --config configs/Citeseer_node_classification_GraphSage_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 95 --config configs/Citeseer_node_classification_GraphSage_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 12 --config configs/Citeseer_node_classification_GraphSage_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 35 --config configs/Citeseer_node_classification_GraphSage_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 41 --config configs/Citeseer_node_classification_GAT_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 95 --config configs/Citeseer_node_classification_GAT_pyg_750k.json &
wait" C-m
tmux send-keys "
python $code --dataset Citeseer --gpu_id 0 --seed 12 --config configs/Citeseer_node_classification_GAT_pyg_750k.json &
python $code --dataset Citeseer --gpu_id 1 --seed 35 --config configs/Citeseer_node_classification_GAT_pyg_750k.json &
wait" C-m
#Pubmed
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 41 --config configs/Pubmed_node_classification_MLP_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 95 --config configs/Pubmed_node_classification_MLP_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 12 --config configs/Pubmed_node_classification_MLP_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 35 --config configs/Pubmed_node_classification_MLP_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 41 --config configs/Pubmed_node_classification_GCN_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 95 --config configs/Pubmed_node_classification_GCN_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 12 --config configs/Pubmed_node_classification_GCN_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 35 --config configs/Pubmed_node_classification_GCN_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 41 --config configs/Pubmed_node_classification_ResGatedGCN_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 95 --config configs/Pubmed_node_classification_ResGatedGCN_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 12 --config configs/Pubmed_node_classification_ResGatedGCN_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 35 --config configs/Pubmed_node_classification_ResGatedGCN_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 41 --config configs/Pubmed_node_classification_MoNet_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 95 --config configs/Pubmed_node_classification_MoNet_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 12 --config configs/Pubmed_node_classification_MoNet_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 35 --config configs/Pubmed_node_classification_MoNet_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 41 --config configs/Pubmed_node_classification_GIN_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 95 --config configs/Pubmed_node_classification_GIN_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 12 --config configs/Pubmed_node_classification_GIN_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 35 --config configs/Pubmed_node_classification_GIN_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 41 --config configs/Pubmed_node_classification_GatedGCN_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 95 --config configs/Pubmed_node_classification_GatedGCN_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 12 --config configs/Pubmed_node_classification_GatedGCN_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 35 --config configs/Pubmed_node_classification_GatedGCN_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 41 --config configs/Pubmed_node_classification_GraphSage_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 95 --config configs/Pubmed_node_classification_GraphSage_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 12 --config configs/Pubmed_node_classification_GraphSage_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 35 --config configs/Pubmed_node_classification_GraphSage_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 41 --config configs/Pubmed_node_classification_GAT_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 95 --config configs/Pubmed_node_classification_GAT_pyg_350k.json &
wait" C-m
tmux send-keys "
python $code --dataset Pubmed --gpu_id 0 --seed 12 --config configs/Pubmed_node_classification_GAT_pyg_350k.json &
python $code --dataset Pubmed --gpu_id 1 --seed 35 --config configs/Pubmed_node_classification_GAT_pyg_350k.json &
wait" C-m







#--dataset SBM_CLUSTER --gpu_id 0 --seed 41 --config configs/SBMs_node_clustering_ResGatedGCN_pyg_100k.json

# tmux send-keys "tmux kill-session -t benchmark" C-m









