




"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
        
# from configs.base import Grid, Config





"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from nets.Planetoid_node_classification.load_net import gnn_model # import GNNs
from data.data import LoadData # import dataset




"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(gpu_id))
        device = torch.device("cuda:"+ str(gpu_id))
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device










"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param



"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    avg_test_acc = []
    avg_train_acc = []
    avg_val_acc = []
    avg_convergence_epochs = []
    t0 = time.time()
    per_epoch_time = []
    
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
    
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""  .format(dataset.name, MODEL_NAME, params, net_params, net_params['total_param']))
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for split_number in range(10):
            training_scores, val_scores, test_scores, epochs = [], [], [], []
            # setting seeds
            random.seed(params['seed'])
            np.random.seed(params['seed'])
            torch.manual_seed(params['seed'])
            if device.type == 'cuda':
                torch.cuda.manual_seed(params['seed'])
            # Mitigate bad random initializations

            train_idx, val_idx, test_idx = dataset.train_idx[split_number], dataset.val_idx[split_number], \
                                           dataset.test_idx[split_number]
            print("Training Nodes: ", len(train_idx))
            print("Validation Nodes: ", len(val_idx))
            print("Test Nodes: ", len(test_idx))
            print("Number of Classes: ", net_params['n_classes'])
            for run in range(3):
                t0_split = time.time()
                print("RUN NUMBER:", split_number, run)
                log_dir = os.path.join(root_log_dir, "RUN_" + str(split_number))
                writer = SummaryWriter(log_dir=log_dir)
                model = gnn_model(MODEL_NAME, net_params)
                model = model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                 factor=params['lr_reduce_factor'],
                                                                 patience=params['lr_schedule_patience'],
                                                                 verbose=True)

                epoch_train_losses, epoch_val_losses = [], []
                epoch_train_accs, epoch_val_accs = [], []

                # import train functions for all other GCNs
                from train.train_Planetoid_node_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

                with tqdm(range(params['epochs']), ncols= 0) as t:
                    for epoch in t:

                        t.set_description('Epoch %d' % epoch)

                        start = time.time()
                        # for all other models common train function
                        epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, dataset, train_idx)

                        epoch_val_loss, epoch_val_acc = evaluate_network(model, device, dataset, val_idx)
                        _, epoch_test_acc = evaluate_network(model, device, dataset, test_idx)

                        epoch_train_losses.append(epoch_train_loss)
                        epoch_val_losses.append(epoch_val_loss)
                        epoch_train_accs.append(epoch_train_acc)
                        epoch_val_accs.append(epoch_val_acc)

                        writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                        writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                        writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                        writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                        writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                        t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                                      train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                      train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                                      test_acc=epoch_test_acc)

                        per_epoch_time.append(time.time()-start)

                        # Saving checkpoint
                        ckpt_dir = os.path.join(root_ckpt_dir, "RUN_" + str(split_number))
                        if not os.path.exists(ckpt_dir):
                            os.makedirs(ckpt_dir)
                        # torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))
                        # it is for save the models.
                        files = glob.glob(ckpt_dir + '/*.pkl')
                        for file in files:
                            epoch_nb = file.split('_')[-1]
                            epoch_nb = int(epoch_nb.split('.')[0])
                            if epoch_nb < epoch-1:
                                os.remove(file)

                        scheduler.step(epoch_val_loss)

                        # it used to test the scripts
                        # if epoch == 1:
                        #     break

                        if optimizer.param_groups[0]['lr'] < params['min_lr']:
                            print("\n!! LR EQUAL TO MIN LR SET.")
                            break

                        # Stop training after params['max_time'] hours
                        if time.time()-t0_split > params['max_time']*3600/10:       # Dividing max_time by 10, since there are 10 runs in TUs
                            print('-' * 89)
                            print("Max_time for one train-val-test split experiment elapsed {:.3f} hours, so stopping".format(params['max_time']/10))
                            break

                _, test_acc = evaluate_network(model, device, dataset, test_idx)
                _, val_acc = evaluate_network(model, device, dataset, val_idx)
                _, train_acc = evaluate_network(model, device, dataset, train_idx)
                training_scores.append(train_acc)
                val_scores.append(val_acc)
                test_scores.append(test_acc)
                epochs.append(epoch)
            training_score = sum(training_scores) / 3
            val_score = sum(val_scores) / 3
            test_score = sum(test_scores) / 3
            epoch_score = sum(epochs) / 3
            avg_val_acc.append(val_score)
            avg_test_acc.append(test_score)
            avg_train_acc.append(training_score)
            avg_convergence_epochs.append(epoch_score)

            print("Test Accuracy [LAST EPOCH]: {:.4f}".format(test_score))
            print("Val Accuracy: {:.4f}".format(val_score))
            print("Train Accuracy [LAST EPOCH]: {:.4f}".format(training_score))
            print("Convergence Time (Epochs): {:.4f}".format(epoch_score))
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
        
    
    print("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time()-t0)/3600))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    print("AVG CONVERGENCE Time (Epochs): {:.4f}".format(np.mean(np.array(avg_convergence_epochs))))
    # Final test accuracy value averaged over 10-fold
    print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}"""          .format(np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100))
    print("\nAll splits Test Accuracies:\n", avg_test_acc)
    print("""\n\n\nFINAL RESULTS\n\nVAL ACCURACY averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(avg_val_acc)) * 100, np.std(avg_val_acc) * 100))
    print("\nAll splits Val Accuracies:\n", avg_val_acc)
    print("""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}"""          .format(np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100))
    print("\nAll splits Train Accuracies:\n", avg_train_acc)

    writer.close()

    """
        Write the results in out/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}\nval ACCURACY averaged: {:.4f} with s.d. {:.4f}\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}\n\n
    Average Convergence Time (Epochs): {:.4f} with s.d. {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\nAll Splits Test Accuracies: {}"""\
          .format(dataset.name, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100,
                  np.mean(np.array(avg_val_acc))*100, np.std(avg_val_acc)*100,
                  np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100,
                  np.mean(avg_convergence_epochs), np.std(avg_convergence_epochs),
               (time.time()-t0)/3600, np.mean(per_epoch_time), avg_test_acc))




def main():    
    """
        USER CONTROLS
    """
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--framework', type=str, default= None, help="Please give a framework to use")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    parser.add_argument('--pos_enc', help="Please give a value for pos_enc")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    # it uses to separate the hyper-parameter, to do
    # model_configurations = Grid(config_file, dataset_name)
    # model_configuration = Config(**model_configurations[0])
    #
    # exp_path = os.path.join(result_folder, f'{model_configuration.exp_name}_assessment')
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    # parameters
    params = config['params']
    params['framework'] = 'pyg' if MODEL_NAME[-3:] == 'pyg' else 'dgl'
    if args.framework is not None:
        params['framework'] = str(args.framework)
    if args.use_node_embedding is not None:
        params['use_node_embedding'] = bool(args.use_node_embedding)
    dataset = LoadData(DATASET_NAME, use_node_embedding = params['use_node_embedding'],framework = params['framework'])
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)



    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']

    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
    if args.pos_enc is not None:
        net_params['pos_enc'] = True if args.pos_enc=='True' else False
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)

    # Planetoid
    net_params['in_dim'] = dataset.dataset[0].x.size(1)
    net_params['n_classes'] = torch.unique(dataset.dataset[0].y,dim=0).size(0)

    if MODEL_NAME == 'DiffPool':
        # calculate assignment dimension: pool_ratio * largest graph's maximum
        # number of nodes  in the dataset
        num_nodes = [dataset.all[i][0].number_of_nodes() for i in range(len(dataset.all))]
        max_num_node = max(num_nodes)
        net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']
        
    if MODEL_NAME == 'RingGNN':
        num_nodes = [dataset.all[i][0].number_of_nodes() for i in range(len(dataset.all))]
        net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))
    
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)

    

    
    
    
    
    
    
main()    
















