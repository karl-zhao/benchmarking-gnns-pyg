#!/bin/bash


############
# Usage
############

# bash script_main_node_classification_arxivs_100k.sh



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
code=main_ogb_node_classification.py
tmux new -s benchmark_ogb-input -d
tmux send-keys "source activate pytorch1.5.0" C-m
tmux send-keys "export CUDA_VISIBLE_DEVICES = 0, 1" C-m
dataset=SBM_CLUSTER
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_MLP_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 95 --config configs/arxiv_node_classification_MLP_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 12 --config configs/arxiv_node_classification_MLP_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 35 --config configs/arxiv_node_classification_MLP_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_MLP_pyg_160k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 95 --config configs/arxiv_node_classification_MLP_pyg_160k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 12 --config configs/arxiv_node_classification_MLP_pyg_160k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 35 --config configs/arxiv_node_classification_MLP_pyg_160k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_MLP_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 95 --config configs/arxiv_node_classification_MLP_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 12 --config configs/arxiv_node_classification_MLP_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 35 --config configs/arxiv_node_classification_MLP_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_MLP_pyg_160k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 95 --config configs/arxiv_node_classification_MLP_pyg_160k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 12 --config configs/arxiv_node_classification_MLP_pyg_160k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 35 --config configs/arxiv_node_classification_MLP_pyg_160k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
#GCN
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_GCN_pyg_90k.json --out_dir out-input/basic/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 95 --config configs/arxiv_node_classification_GCN_pyg_90k.json --out_dir out-input/basic/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 12 --config configs/arxiv_node_classification_GCN_pyg_90k.json --out_dir out-input/basic/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 35 --config configs/arxiv_node_classification_GCN_pyg_90k.json --out_dir out-input/basic/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_GCN_pyg_160k.json --out_dir out-input/basic/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 95 --config configs/arxiv_node_classification_GCN_pyg_160k.json --out_dir out-input/basic/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 12 --config configs/arxiv_node_classification_GCN_pyg_160k.json --out_dir out-input/basic/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 35 --config configs/arxiv_node_classification_GCN_pyg_160k.json --out_dir out-input/basic/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_GCN_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 95 --config configs/arxiv_node_classification_GCN_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 12 --config configs/arxiv_node_classification_GCN_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 35 --config configs/arxiv_node_classification_GCN_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_GCN_pyg_160k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 95 --config configs/arxiv_node_classification_GCN_pyg_160k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 12 --config configs/arxiv_node_classification_GCN_pyg_160k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 35 --config configs/arxiv_node_classification_GCN_pyg_160k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_GCN_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 95 --config configs/arxiv_node_classification_GCN_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 12 --config configs/arxiv_node_classification_GCN_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 35 --config configs/arxiv_node_classification_GCN_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_GCN_pyg_160k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 95 --config configs/arxiv_node_classification_GCN_pyg_160k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 12 --config configs/arxiv_node_classification_GCN_pyg_160k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 35 --config configs/arxiv_node_classification_GCN_pyg_160k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
#GAT
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_GAT_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 95 --config configs/arxiv_node_classification_GAT_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 12 --config configs/arxiv_node_classification_GAT_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 35 --config configs/arxiv_node_classification_GAT_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_GAT_pyg_160k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 95 --config configs/arxiv_node_classification_GAT_pyg_160k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 12 --config configs/arxiv_node_classification_GAT_pyg_160k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 35 --config configs/arxiv_node_classification_GAT_pyg_160k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_GAT_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 95 --config configs/arxiv_node_classification_GAT_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 12 --config configs/arxiv_node_classification_GAT_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 35 --config configs/arxiv_node_classification_GAT_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_GAT_pyg_160k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 95 --config configs/arxiv_node_classification_GAT_pyg_160k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 12 --config configs/arxiv_node_classification_GAT_pyg_160k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 1 --seed 35 --config configs/arxiv_node_classification_GAT_pyg_160k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m

#Products
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 41 --config configs/products_node_classification_MLP_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 95 --config configs/products_node_classification_MLP_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 12 --config configs/products_node_classification_MLP_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 35 --config configs/products_node_classification_MLP_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 41 --config configs/products_node_classification_MLP_pyg_300k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 95 --config configs/products_node_classification_MLP_pyg_300k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 12 --config configs/products_node_classification_MLP_pyg_300k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 35 --config configs/products_node_classification_MLP_pyg_300k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 41 --config configs/products_node_classification_MLP_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 95 --config configs/products_node_classification_MLP_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 12 --config configs/products_node_classification_MLP_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 35 --config configs/products_node_classification_MLP_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 41 --config configs/products_node_classification_MLP_pyg_300k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 95 --config configs/products_node_classification_MLP_pyg_300k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 12 --config configs/products_node_classification_MLP_pyg_300k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 35 --config configs/products_node_classification_MLP_pyg_300k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
#GCN
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 41 --config configs/products_node_classification_GCN_pyg_90k.json --out_dir out-input/basic/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 95 --config configs/products_node_classification_GCN_pyg_90k.json --out_dir out-input/basic/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 12 --config configs/products_node_classification_GCN_pyg_90k.json --out_dir out-input/basic/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 35 --config configs/products_node_classification_GCN_pyg_90k.json --out_dir out-input/basic/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 41 --config configs/products_node_classification_GCN_pyg_300k.json --out_dir out-input/basic/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 95 --config configs/products_node_classification_GCN_pyg_300k.json --out_dir out-input/basic/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 12 --config configs/products_node_classification_GCN_pyg_300k.json --out_dir out-input/basic/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 35 --config configs/products_node_classification_GCN_pyg_300k.json --out_dir out-input/basic/ &
wait" C-m
#GCN
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 41 --config configs/products_node_classification_GCN_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 95 --config configs/products_node_classification_GCN_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 12 --config configs/products_node_classification_GCN_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 35 --config configs/products_node_classification_GCN_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 41 --config configs/products_node_classification_GCN_pyg_300k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 95 --config configs/products_node_classification_GCN_pyg_300k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 12 --config configs/products_node_classification_GCN_pyg_300k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 35 --config configs/products_node_classification_GCN_pyg_300k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 41 --config configs/products_node_classification_GCN_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 95 --config configs/products_node_classification_GCN_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 12 --config configs/products_node_classification_GCN_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 35 --config configs/products_node_classification_GCN_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 41 --config configs/products_node_classification_GCN_pyg_300k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 95 --config configs/products_node_classification_GCN_pyg_300k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 12 --config configs/products_node_classification_GCN_pyg_300k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 35 --config configs/products_node_classification_GCN_pyg_300k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
#GAT
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 41 --config configs/products_node_classification_GAT_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 95 --config configs/products_node_classification_GAT_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 12 --config configs/products_node_classification_GAT_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 35 --config configs/products_node_classification_GAT_pyg_90k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 41 --config configs/products_node_classification_GAT_pyg_300k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 95 --config configs/products_node_classification_GAT_pyg_300k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 12 --config configs/products_node_classification_GAT_pyg_300k.json --use_node_embedding --out_dir out-input/node_embedding/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 35 --config configs/products_node_classification_GAT_pyg_300k.json --use_node_embedding --out_dir out-input/node_embedding/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 41 --config configs/products_node_classification_GAT_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 95 --config configs/products_node_classification_GAT_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 12 --config configs/products_node_classification_GAT_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 35 --config configs/products_node_classification_GAT_pyg_90k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 41 --config configs/products_node_classification_GAT_pyg_300k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 95 --config configs/products_node_classification_GAT_pyg_300k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 0 --seed 12 --config configs/products_node_classification_GAT_pyg_300k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
python main_ogb_node_classification.py --dataset ogbn-products --gpu_id 1 --seed 35 --config configs/products_node_classification_GAT_pyg_300k.json --pos_enc --pos_enc_dim 20 --out_dir out-input/posenc/ &
wait" C-m

#modify the gcn results
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-mag --gpu_id 0 --seed 41 --config configs/mag_node_classification_GCN_pyg_130k.json &
python main_ogb_node_classification.py --dataset ogbn-mag --gpu_id 1 --seed 95 --config configs/mag_node_classification_GCN_pyg_130k.json &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-mag --gpu_id 0 --seed 12 --config configs/mag_node_classification_GCN_pyg_130k.json &
python main_ogb_node_classification.py --dataset ogbn-mag --gpu_id 1 --seed 35 --config configs/mag_node_classification_GCN_pyg_130k.json &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-mag --gpu_id 0 --seed 41 --config configs/mag_node_classification_GCN_pyg_330k.json &
python main_ogb_node_classification.py --dataset ogbn-mag --gpu_id 1 --seed 95 --config configs/mag_node_classification_GCN_pyg_330k.json &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-mag --gpu_id 0 --seed 12 --config configs/mag_node_classification_GCN_pyg_330k.json &
python main_ogb_node_classification.py --dataset ogbn-mag --gpu_id 1 --seed 35 --config configs/mag_node_classification_GCN_pyg_330k.json &
wait" C-m

tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-proteins --gpu_id 0 --seed 41 --config configs/proteins_node_classification_GCN_pyg_90k.json &
python main_ogb_node_classification.py --dataset ogbn-proteins --gpu_id 1 --seed 95 --config configs/proteins_node_classification_GCN_pyg_90k.json &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-proteins --gpu_id 0 --seed 12 --config configs/proteins_node_classification_GCN_pyg_90k.json &
python main_ogb_node_classification.py --dataset ogbn-proteins --gpu_id 1 --seed 35 --config configs/proteins_node_classification_GCN_pyg_90k.json &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-proteins --gpu_id 0 --seed 41 --config configs/proteins_node_classification_GCN_pyg_300k.json &
python main_ogb_node_classification.py --dataset ogbn-proteins --gpu_id 1 --seed 95 --config configs/proteins_node_classification_GCN_pyg_300k.json &
wait" C-m
tmux send-keys "
python main_ogb_node_classification.py --dataset ogbn-proteins --gpu_id 0 --seed 12 --config configs/proteins_node_classification_GCN_pyg_300k.json &
python main_ogb_node_classification.py --dataset ogbn-proteins --gpu_id 1 --seed 35 --config configs/proteins_node_classification_GCN_pyg_300k.json &
wait" C-m