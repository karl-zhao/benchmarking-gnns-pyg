# Reproducibility


<br>

## 1. Usage


<br>

### 1.1 In terminal

```
# Run the main file (at the root of the project)
python main_molecules_graph_regression.py --dataset ZINC --config 'configs/molecules_graph_regression_GatedGCN_ZINC_100k.json' # for CPU
python main_molecules_graph_regression.py --dataset ZINC --gpu_id 0 --config 'configs/molecules_graph_regression_GatedGCN_ZINC_100k.json' # for GPU
```
The training and network parameters for each dataset and network is stored in a json file in the [`configs/`](../configs) directory.

<br>

### 1.2 run the scripts
Run the [script](../scripts/),you can see the scripts for detail.

if want to use the node2vec embedding, first run [script](../data/), node2vec, for create the embeddings.

<br>

## 2. Output, checkpoints and visualizations

Output results are located in the folder defined by the variable `out_dir` in the corresponding config file (eg. [`configs/molecules_graph_regression_GatedGCN_ZINC_100k.json`](../configs/molecules_graph_regression_GatedGCN_ZINC_100k.json) file).  

If `out_dir = 'out/molecules_graph_regression/'`, then 

#### 2.1 To see checkpoints and results
1. Go to`out/molecules_graph_regression/results` to view all result text files.
2. Directory `out/molecules_graph_regression/checkpoints` contains model checkpoints.

#### 2.2 To see the training logs in Tensorboard on local machine
1. Go to the logs directory, i.e. `out/molecules_graph_regression/logs/`.
2. Run the commands
```
source activate benchmark_gnn
tensorboard --logdir='./' --port 6006
```
3. Open `http://localhost:6006` in your browser. Note that the port information (here 6006 but it may change) appears on the terminal immediately after starting tensorboard.


#### 2.3 To see the training logs in Tensorboard on remote machine
1. Go to the logs directory, i.e. `out/molecules_graph_regression/logs/`.
2. Run the [script](../scripts/TensorBoard/script_tensorboard.sh) with `bash script_tensorboard.sh`.
3. On your local machine, run the command `ssh -N -f -L localhost:6006:localhost:6006 user@xx.xx.xx.xx`.
4. Open `http://localhost:6006` in your browser. Note that `user@xx.xx.xx.xx` corresponds to your user login and the IP of the remote machine.



<br>


<br><br><br>