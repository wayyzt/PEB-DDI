from datetime import datetime
import time 
import argparse
import torch
from get_args import config
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import models_t
from data_preprocessing_t import DrugDataset, DrugDataLoader
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

######################### Parameters ######################
dataset_name = config['dataset_name']
pkl_name = config[dataset_name]["transductive_pkl_dir"]
params = config['params']
lr = params['lr']
n_epochs = params['n_epochs']
batch_size = params['batch_size']
weight_decay = params['weight_decay']
neg_samples = params['neg_samples']
data_size_ratio = params['data_size_ratio']
device = 'cuda:0' if torch.cuda.is_available() and params['use_cuda'] else 'cpu'
print(dataset_name, params)
n_atom_feats = 55
rel_total = 86
kge_dim = 128
######################### Dataset ######################
def split_train_valid(data, fold, val_ratio=0.2):
    data = np.array(data)
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=data, y=data[:, 2])))
    train_tup = data[train_index]
    val_tup = data[val_index]
    train_tup = [(tup[0],tup[1],int(tup[2]))for tup in train_tup ]
    val_tup = [(tup[0],tup[1],int(tup[2]))for tup in val_tup ]

    return train_tup, val_tup

if 'drugbank' not in dataset_name:
    df_ddi_train = pd.read_csv(config[dataset_name]["trans_ddi_train"])
    df_ddi_test = pd.read_csv(config[dataset_name]["trans_ddi_test"])
    df_ddi_valid= pd.read_csv(config[dataset_name]["trans_ddi_valid"])

    train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['drugbank_id_1'], df_ddi_train['drugbank_id_2'], df_ddi_train['label'])]
    val_tup = [(h, t, r) for h, t, r in zip(df_ddi_valid['drugbank_id_1'], df_ddi_valid['drugbank_id_2'], df_ddi_valid['label'])]
    test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['drugbank_id_1'], df_ddi_test['drugbank_id_2'], df_ddi_test['label'])]
else:
    df_ddi_train = pd.read_csv(config[dataset_name]["trans_ddi_train"])
    df_ddi_test = pd.read_csv(config[dataset_name]["trans_ddi_test"])

    train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
    train_tup, val_tup = split_train_valid(train_tup,2, val_ratio=0.2)
    test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False)
test_data = DrugDataset(test_tup, disjoint_split=False)

print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=2)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size *3,num_workers=2)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size *3,num_workers=2)

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

def test(test_data_loader,model):
    test_probas_pred = []
    test_ground_truth = []
    with torch.no_grad():
        for batch in test_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            test_probas_pred.append(probas_pred)
            test_ground_truth.append(ground_truth)
        test_probas_pred = np.concatenate(test_probas_pred)
        test_ground_truth = np.concatenate(test_ground_truth)
        test_acc, test_auc_roc, test_f1, test_precision,test_recall,test_int_ap, test_ap = do_compute_metrics(test_probas_pred, test_ground_truth)
    print('\n')
    print('============================== Test Result ==============================')
    print(f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f},test_f1: {test_f1:.4f},test_precision:{test_precision:.4f}')
    print(f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')


model = models_t.MVN_DDI([n_atom_feats, 2048, 200], 17, kge_dim, kge_dim, rel_total, [64,64,64,64], [2, 2, 2, 2], 64, 0.0)
model.load_state_dict(torch.load(pkl_name))
model.to(device=device)
test(test_data_loader,model)


