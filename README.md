

# Benchmarking Graph Neural Networks for node classification

<br>
Note: This repository modified from graphdeeplearning/benchmarking-gnns.
We refactor the code using the pyg framework and add the planetoid and ogb datasets to make the node classification.

## 1. Benchmark installation

1. Setup Python environment for GPU

```
git clone https://https://github.com/karl-zhao/benchmarking-gnns-pyg.git
cd benchmarking-gnns-pyg
# Install python environment
conda env create -f environment_gpu.yml 
# Activate environment
conda activate pytorch1.5.0
```

<br>

## 2. Download datasets

All the datasets can be downloaded automatically except SBMs. For the SBMs, run the [data/SBMs/generate_SBM_CLUSTER.ipynb](data/SBMs/generate_SBM_CLUSTER.ipynb) and [data/SBMs/generate_SBM_PATTERN.ipynb](data/SBMs/generate_SBM_PATTERN.ipynb) first to generate the dataset.




<br>

## 3. Reproducibility 

##### 3.1 in terminal
```
# Run the main file (at the root of the project)
python main_arxiv_node_classification.py --dataset ogbn-arxiv --gpu_id 0 --seed 41 --config configs/arxiv_node_classification_GAT_pyg_90k.json # for GPU
```
It will first download the datasets and then train the model.

The training and network parameters for each dataset and network is stored in a json file in the [`configs/`](../configs) directory.

##### 3.2 run the scripts
Run the [script](scripts/),you can see the scripts for detail.
```
# Run the scripts (at the root of the project)
bash scripts/ogbs/script_main_node_classification_arxivs_100k.sh # for GPU
```

If want to use the node2vec embedding, first run the main function to download the datasets. Then run [node2vec_***.py](data/) for create the embeddings.

##### 3.3 Output, checkpoints and visualizations

Output results are located in the folder defined by the variable `out_dir` in the corresponding config file (eg. [`configs/arxiv_node_classification_GAT_pyg_90k.json`](../configs/arxiv_node_classification_GAT_pyg_90k.json) file). 
ourdir also can change using `--out_dir `

##### 3.4 To see checkpoints and results
1. Go to`out/ogb_node_classification/results` to view all result text files.
2. Directory `out/ogb_node_classification/checkpoints` contains model checkpoints.

##### 3.5 To see the training logs in Tensorboard on local machine
1. Go to the logs directory, i.e. `out/molecules_graph_regression/logs/`.
2. Run the commands
```
source activate benchmark_gnn
tensorboard --logdir='./' --port 6006
```
3. Open `http://localhost:6006` in your browser. Note that the port information (here 6006 but it may change) appears on the terminal immediately after starting tensorboard.


##### 3.6 To see the training logs in Tensorboard on remote machine
1. Go to the logs directory, i.e. `out/molecules_graph_regression/logs/`.
2. Run the [script](script_tensorboard.sh) with `bash script_tensorboard.sh`.
3. On your local machine, run the command `ssh -N -f -L localhost:6006:localhost:6006 user@xx.xx.xx.xx`.
4. Open `http://localhost:6006` in your browser. Note that `user@xx.xx.xx.xx` corresponds to your user login and the IP of the remote machine.
<br>


<br><br><br>

### Citation


```
@misc{zhao2020pipeline,
      title={A pipeline for fair comparison of graph neural networks in node classification tasks}, 
      author={Wentao Zhao and Dalin Zhou and Xinguo Qiu and Wei Jiang},
      year={2020},
      eprint={2012.10619},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
``` 
