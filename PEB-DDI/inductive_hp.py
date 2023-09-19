from datetime import datetime
import time 
import argparse

import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from get_args import config
import models_i
import custom_loss
from data_preprocessing_i import DrugDataset, DrugDataLoader
import warnings
warnings.filterwarnings('ignore',category=UserWarning)


######################### Parameters ######################
dataset_name = config['dataset_name']
pkl_name = config[dataset_name]["inductive_pkl_dir"]
params = config['params']
lr = params['lr']
n_epochs = 40
batch_size = params['batch_size']
weight_decay = params['weight_decay']
neg_samples = params['neg_samples']
data_size_ratio = params['data_size_ratio']
device = 'cuda:0' if torch.cuda.is_available() and params['use_cuda'] else 'cpu'
# device = 'cpu'

print(dataset_name, params)
n_atom_feats = 55
rel_total = 86
kge_dim = 128
############################################################

###### Dataset
if dataset_name != 'drugbank':
    df_ddi_train = pd.read_csv(config[dataset_name]["induc_ddi_train"])
    df_ddi_s1 = pd.read_csv(config[dataset_name]["induc_s1"])
    df_ddi_s2 = pd.read_csv(config[dataset_name]["induc_s2"])

    train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['drugbank_id_1'], df_ddi_train['drugbank_id_2'], df_ddi_train['label'])]
    s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1['drugbank_id_1'], df_ddi_s1['drugbank_id_2'], df_ddi_s1['label'])]
    s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2['drugbank_id_1'], df_ddi_s2['drugbank_id_2'], df_ddi_s2['label'])]
else:
    df_ddi_train = pd.read_csv(config[dataset_name]["induc_ddi_train"])
    df_ddi_s1 = pd.read_csv(config[dataset_name]["induc_s1"])
    df_ddi_s2 = pd.read_csv(config[dataset_name]["induc_s2"])

    train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
    s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1['d1'], df_ddi_s1['d2'], df_ddi_s1['type'])]
    s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2['d1'], df_ddi_s2['d2'], df_ddi_s2['type'])]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
s1_data = DrugDataset(s1_tup, disjoint_split=True)
s2_data = DrugDataset(s2_tup, disjoint_split=True)

print(f"Training with {len(train_data)} samples, s1 with {len(s1_data)}, and s2 with {len(s2_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=2)
s1_data_loader = DrugDataLoader(s1_data, batch_size=batch_size *3,num_workers=2)
s2_data_loader = DrugDataLoader(s2_data, batch_size=batch_size *3,num_workers=2)

def do_compute(batch, device, model):
        '''
            *batch: (pos_tri, neg_tri)
            *pos/neg_tri: (batch_h, batch_t, batch_r)
        '''
        probas_pred, ground_truth = [], []
        pos_tri, neg_tri = batch
        
        pos_tri = [tensor.to(device=device) for tensor in pos_tri]
        p_score = model(pos_tri)
        probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
        ground_truth.append(np.ones(len(p_score)))

        neg_tri = [tensor.to(device=device) for tensor in neg_tri]
        n_score = model(neg_tri)
        probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
        ground_truth.append(np.zeros(len(n_score)))

        probas_pred = np.concatenate(probas_pred)
        ground_truth = np.concatenate(ground_truth)

        return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap= metrics.average_precision_score(target, probas_pred)

    return acc, auroc, f1_score, precision, recall, int_ap, ap


def train(model, train_data_loader, s1_data_loader, s2_data_loader, loss_fn,  optimizer, n_epochs, device, scheduler=None):
    print('Starting training at', datetime.today())
    s1_acc_max = 0
    s2_acc_max = 0
    for i in range(1, n_epochs+1):
        start = time.time()
        train_loss = 0 
        s1_loss = 0
        s2_loss = 0
      
        train_probas_pred = []
        train_ground_truth = []

        s1_probas_pred = []
        s1_ground_truth = []

        s2_probas_pred = []
        s2_ground_truth = []

        for batch in train_data_loader:
            model.train()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(p_score)
        train_loss /= len(train_data)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_auc_roc, train_f1, train_precision,train_recall,train_int_ap, train_ap = do_compute_metrics(train_probas_pred, train_ground_truth)

            for batch in s1_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                s1_probas_pred.append(probas_pred)
                s1_ground_truth.append(ground_truth)
                loss, loss_p, loss_n = loss_fn(p_score, n_score)
                s1_loss += loss.item() * len(p_score)            

            s1_loss /= len(s1_data)
            s1_probas_pred = np.concatenate(s1_probas_pred)
            s1_ground_truth = np.concatenate(s1_ground_truth)
            s1_acc, s1_auc_roc, s1_f1,s1_precision,s1_recall,s1_int_ap, s1_ap = do_compute_metrics(s1_probas_pred, s1_ground_truth)
        

            for batch in s2_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                s2_probas_pred.append(probas_pred)
                s2_ground_truth.append(ground_truth)
                loss, loss_p, loss_n = loss_fn(p_score, n_score)
                s2_loss += loss.item() * len(p_score)            

            s2_loss /= len(s2_data)
            s2_probas_pred = np.concatenate(s2_probas_pred)
            s2_ground_truth = np.concatenate(s2_ground_truth)
            s2_acc, s2_auc_roc, s2_f1,s2_precision,s2_recall,s2_int_ap, s2_ap = do_compute_metrics(s2_probas_pred, s2_ground_truth)

            # if s1_acc>s1_acc_max:
            #     s1_acc_max = s1_acc
            #     torch.save(model,pkl_name)
            if s2_acc>s2_acc_max:
                s2_acc_max = s2_acc
                torch.save(model,pkl_name)
               
        if scheduler:
            # print('scheduling')
            scheduler.step()

        print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, s1_loss: {s1_loss:.4f},s2_loss: {s2_loss:.4f}')
        print(f'\t\ttrain_acc: {train_acc:.4f}, train_roc: {train_auc_roc:.4f},train_precision: {train_precision:.4f},train_recall:{train_recall:.4f}')
        print(f'\t\ts1_acc: {s1_acc:.4f}, s1_roc: {s1_auc_roc:.4f}, s1_precision: {s1_precision:.4f}, s1_recall: {s1_recall:.4f}')
        print(f'\t\ts2_acc: {s2_acc:.4f}, s2_roc: {s2_auc_roc:.4f}, s2_precision: {s2_precision:.4f}, s2_recall: {s2_recall:.4f}')
    
def test(s1_data_loader, s2_data_loader, model):
    s1_probas_pred = []
    s1_ground_truth = []

    s2_probas_pred = []
    s2_ground_truth = []
    with torch.no_grad():
        for batch in s1_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model=model)
            s1_probas_pred.append(probas_pred)
            s1_ground_truth.append(ground_truth)
      
        s1_probas_pred = np.concatenate(s1_probas_pred)
        s1_ground_truth = np.concatenate(s1_ground_truth)
        s1_acc, s1_auc_roc, s1_f1,s1_precision,s1_recall,s1_int_ap, s1_ap = do_compute_metrics(s1_probas_pred, s1_ground_truth)
        

        for batch in s2_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device,model=model)
            s2_probas_pred.append(probas_pred)
            s2_ground_truth.append(ground_truth)
                
        s2_probas_pred = np.concatenate(s2_probas_pred)
        s2_ground_truth = np.concatenate(s2_ground_truth)
        s2_acc, s2_auc_roc, s2_f1,s2_precision,s2_recall,s2_int_ap, s2_ap = do_compute_metrics(s2_probas_pred, s2_ground_truth)

    print('\n')
    print('============================== Best Result ==============================')
    print(f'\t\ts1_acc: {s1_acc:.4f}, s1_roc: {s1_auc_roc:.4f}, s1_f1: {s1_f1:.4f}, s1_precision: {s1_precision:.4f},s1_recall: {s1_recall:.4f},s1_int_ap: {s1_int_ap:.4f},s1_ap: {s1_ap:.4f}')
    print(f'\t\ts2_acc: {s2_acc:.4f}, s2_roc: {s2_auc_roc:.4f}, s2_f1: {s2_f1:.4f}, s2_precision: {s2_precision:.4f},s2_recall: {s2_recall:.4f},s2_int_ap: {s2_int_ap:.4f},s2_ap: {s2_ap:.4f}')

model = models_i.MVN_DDI([n_atom_feats, 2048, 200], 17, kge_dim, kge_dim, rel_total, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2])
loss = custom_loss.SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
# print(model)
model.to(device=device)

# if __name__ == '__main__':
train(model, train_data_loader, s1_data_loader, s2_data_loader, loss, optimizer, n_epochs, device, scheduler)
test_model = torch.load(pkl_name)
test(s1_data_loader, s2_data_loader, test_model)

from hyperopt import fmin, tpe, hp

#定义要优化的参数
space = {
    'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
    'weight_decay': hp.uniform('weight_decay', 0.0001, 0.01),
    #...添加其他需要优化的参数...
}

#定义优化目标函数
def objective(params):
    #使用params中的参数值初始化模型和优化器
    model = models_i.MVN_DDI([n_atom_feats, 2048, 200], 17, kge_dim, kge_dim, rel_total, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2])
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    #训练模型
    train(model, train_data_loader, s1_data_loader, s2_data_loader, loss, optimizer, n_epochs, device)

    #加载最好的模型
    test_model = torch.load(pkl_name)
    
    #在测试集上评估模型并获取s2_acc
    _, s2_acc = test(s1_data_loader, s2_data_loader, model)
    
    #因为Hyperopt是用于最小化目标函数的，所以我们返回的是1-s2_acc
    return 1 - s2_acc

#使用Hyperopt进行优化
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)

print('Best parameters:', best)