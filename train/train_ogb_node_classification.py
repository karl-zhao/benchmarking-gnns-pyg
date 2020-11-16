"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl
from tqdm import tqdm
from train.metrics import accuracy_SBM as accuracy
from train.metrics import accuracy_ogb
from ogb.nodeproppred import Evaluator

"""
    For GCNs
"""
def train_epoch(model, optimizer, device, train_loader, epoch=None):
    model.train()

    # pbar = tqdm(total=len(train_loader))
    # pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        try:
            batch_pos_enc = data.pos_enc.to(device)
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            batch_scores = model.forward(data.x, data.edge_index, data.edge_attr,batch_pos_enc)
        except:
            batch_scores = model(data.x, data.edge_index, data.edge_attr)
        loss = model.loss(batch_scores[data.train_mask], data.y.view(-1)[data.train_mask]).to(torch.float)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

    #     pbar.update(1)
    #
    # pbar.close()

    return total_loss / total_examples
    # model.train()
    #
    # # for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
    # batch_x = dataset.x.to(device)
    # batch_e = dataset.edge_attr.to(device)
    # # batch_e = dataset.edge_attr
    # batch_labels = dataset.y.long().to(device)
    # edge_index = dataset.edge_index.long().to(device)
    # train_idx = train_idx.to(device)
    #
    # optimizer.zero_grad()
    # batch_scores = model.forward(batch_x, edge_index, batch_e)[train_idx]
    # loss = model.loss(batch_scores, batch_labels.view(-1)[train_idx]).to(torch.float)
    # loss.backward()
    # optimizer.step()
    # epoch_loss = loss.detach().item()
    #
    # return epoch_loss


def train_epoch_arxiv(model, optimizer, device, dataset, train_idx):
    model.train()

    # for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
    batch_x = dataset.x.to(device)
    batch_e = dataset.edge_attr.to(device)
    # batch_e = dataset.edge_attr
    batch_labels = dataset.y.long().to(device)
    edge_index = dataset.edge_index.long().to(device)
    train_idx = train_idx.to(device)

    optimizer.zero_grad()
    try:
        batch_pos_enc = dataset.pos_enc.to(device)
        sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        batch_scores = model.forward(batch_x, edge_index, batch_e, batch_pos_enc)[train_idx]
    except:
        batch_scores = model.forward(batch_x, edge_index, batch_e)[train_idx]
    loss = model.loss(batch_scores, batch_labels.view(-1)[train_idx]).to(torch.float)
    loss.backward()
    optimizer.step()
    epoch_loss = loss.detach().item()

    return epoch_loss

def train_epoch_proteins(model, optimizer, device, train_loader, epoch=None):
    model.train()

    # pbar = tqdm(total=len(train_loader))
    # pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        try:
            batch_pos_enc = data.pos_enc.to(device)
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            out = model.forward(data.x, data.edge_index, data.edge_attr,batch_pos_enc)
        except:
            out = model(data.x, data.edge_index, data.edge_attr)
        loss = model.loss_proteins(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

    #     pbar.update(1)
    #
    # pbar.close()

    return total_loss / total_examples

@torch.no_grad()
def evaluate_network(model, device, test_loader, evaluator, epoch):
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    # pbar = tqdm(total=len(test_loader))
    # pbar.set_description(f'Evaluating epoch: {epoch:04d}')
    total_loss = total_examples = 0
    for data in test_loader:
        data = data.to(device)
        try:
            batch_pos_enc = data.pos_enc.to(device)
            out = model.forward(data.x, data.edge_index.long(), data.edge_attr, batch_pos_enc)
        except:
            out = model.forward(data.x, data.edge_index.long(), data.edge_attr)
        # out = model(data.x, data.edge_index.long(), data.edge_attr)

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].argmax(dim=-1, keepdim=True).cpu())
        loss = model.loss(out[data.valid_mask], data.y.view(-1)[data.valid_mask])
        total_loss += float(loss) * int(data.valid_mask.sum())
        total_examples += int(data.valid_mask.sum())
    #     pbar.update(1)
    # pbar.close()

    train_acc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['acc']
    test_acc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['acc']

    return train_acc, valid_acc, test_acc, total_loss / total_examples

    # model.train()
    #
    # # for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
    # batch_x = dataset.x.to(device)
    # batch_e = dataset.edge_attr.to(device)
    # batch_labels = dataset.y.long().to(device)
    # edge_index = dataset.edge_index.long().to(device)
    # train_idx = train_idx.to(device)
    #
    # optimizer.zero_grad()
    # batch_scores = model.forward(batch_x, edge_index, batch_e)[train_idx]
    # loss = model.loss_proteins(batch_scores, batch_labels[train_idx]).to(torch.float)
    # loss.backward()
    # optimizer.step()
    # epoch_loss = loss.detach().item()
    #
    # return epoch_loss

@torch.no_grad()
def evaluate_network_arxiv(model, device, dataset, evaluator):

    model.eval()
    batch_x = dataset.dataset[0].x.to(device)
    y_true = dataset.dataset[0].y.long().to(device)
    split_idx = dataset.split_idx
    batch_e = dataset.dataset[0].edge_attr.to(device)
    edge_index = dataset.dataset[0].edge_index.long().to(device)
    try:
        batch_pos_enc = dataset.dataset[0].pos_enc.to(device)
        batch_scores = model.forward(batch_x, edge_index, batch_e, batch_pos_enc)
    except:
        batch_scores = model.forward(batch_x, edge_index, batch_e)
    # batch_scores = model.forward(batch_x, edge_index, batch_e)
    loss = model.loss(batch_scores[split_idx['valid']], y_true.view(-1)[split_idx['valid']]).to(torch.float)
    epoch_valid_loss = loss.detach().item()
    # y_pred = batch_scores
    y_pred = batch_scores.argmax(dim=-1, keepdim=True)
    # y_true = y_true.view(-1, 1)
    y_true = y_true.view(-1, 1)
    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc, epoch_valid_loss

@torch.no_grad()
def evaluate_network_proteins(model, device, test_loader, evaluator, epoch = None):
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    # pbar = tqdm(total=len(test_loader))
    # pbar.set_description(f'Evaluating epoch: {epoch:04d}')
    total_loss = total_examples = 0
    for data in test_loader:
        data = data.to(device)
        try:
            batch_pos_enc = data.pos_enc.to(device)
            out = model.forward(data.x, data.edge_index, data.edge_attr, batch_pos_enc)
        except:
            out = model.forward(data.x, data.edge_index, data.edge_attr)
        # out = model(data.x, data.edge_index, data.edge_attr)

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())
        loss = model.loss_proteins(out[data.valid_mask], data.y[data.valid_mask])
        total_loss += float(loss) * int(data.valid_mask.sum())
        total_examples += int(data.valid_mask.sum())
    #     pbar.update(1)
    # pbar.close()

    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['rocauc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc, total_loss / total_examples
    #
    # model.eval()
    # batch_x = dataset.dataset[0].x.to(device)
    # y_true = dataset.dataset[0].y.long().to(device)
    # split_idx = dataset.split_idx
    # batch_e = dataset.dataset[0].edge_attr.to(device)
    # edge_index = dataset.dataset[0].edge_index.long().to(device)
    #
    # batch_scores = model.forward(batch_x, edge_index, batch_e)
    # loss = model.loss_proteins(batch_scores[split_idx['valid']], y_true[split_idx['valid']]).to(torch.float)
    # epoch_valid_loss = loss.detach().item()
    # y_pred = batch_scores
    # # y_pred = batch_scores.argmax(dim=-1, keepdim=True)
    # # y_true = y_true.view(-1, 1)
    # train_acc = evaluator.eval({
    #     'y_true': y_true[split_idx['train']],
    #     'y_pred': y_pred[split_idx['train']],
    # })['rocauc']
    # valid_acc = evaluator.eval({
    #     'y_true': y_true[split_idx['valid']],
    #     'y_pred': y_pred[split_idx['valid']],
    # })['rocauc']
    # test_acc = evaluator.eval({
    #     'y_true': y_true[split_idx['test']],
    #     'y_pred': y_pred[split_idx['test']],
    # })['rocauc']
    #
    # return train_acc, valid_acc, test_acc, epoch_valid_loss



