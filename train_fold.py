import librosa

import numpy as np
import pandas as pd
import random

from sklearn.model_selection import KFold, StratifiedKFold

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F


import torch
import os

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from sklearn.metrics import roc_auc_score


from models.Conv import ResNetTransformerModel

def train_fold(train_dataset, CONFIG):

    criterion = nn.MSELoss().to(device)
    
    fold = KFold(n_splits = CONFIG.N_FOLD, shuffle = True, random_state = CONFIG.SEED)

    for f, (train_idx, val_idx) in enumerate(fold.split(train_dataset)) : 
        best_val_score = 0
        best_val_loss = 2

        #model = Model(CONFIG.conf_dic['model_config'] ,CONFIG)
        model = ResNetTransformerModel(1, 2, CONFIG)
        model.to(device)
        train_ds = Subset(train_dataset, train_idx)
        val_ds = Subset(train_dataset, val_idx)

        optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.00001)
        warmup_ratio = 0.1
        t_total = len(train_ds) * CONFIG.N_EPOCHS
        warmup_step = int(t_total * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_step, num_training_steps = t_total)

        train_dl = DataLoader(train_ds, batch_size = CONFIG.BATCH_SIZE , shuffle = True, num_workers = 4)
        val_dl = DataLoader(val_ds, batch_size = CONFIG.BATCH_SIZE, shuffle = False, num_workers = 4)

        for epoch in range(1, CONFIG.N_EPOCHS+1) :
            model.train()
            train_loss = []
            for features, labels in tqdm(iter(train_dl)):
                features = features.float().to(device)
                labels = labels.float().to(device)
                optimizer.zero_grad()
                #last_hidden, output = model(features)
                output = model(features)
                loss = criterion(output, labels)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss.append(loss.item())
                        
            _val_loss, _val_score = validation(model, criterion, val_dl, device)
            _train_loss = np.mean(train_loss)
            print(f'FOLD [{f + 1}] Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}]')
                
            if best_val_loss > _val_loss:
                best_val_loss = _val_loss
                torch.save(model.state_dict(), f'./fold_{f}_best.pth')
    

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score
    
def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in tqdm(iter(val_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            probs = model(features)
            loss = criterion(probs, labels)

            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        _val_loss = np.mean(val_loss)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Calculate AUC score
        auc_score = multiLabel_AUC(all_labels, all_probs)
    
    return _val_loss, auc_score