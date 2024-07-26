import librosa

import numpy as np
import pandas as pd
import random

from sklearn.model_selection import KFold

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
from models.ResTransformer import ResNetTransformerModel
from dataset.spectrogram_dataset import collate_fn

import torch
import os

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from sklearn.metrics import roc_auc_score

from conf.configure import Config
CONFIG = Config()

def train(model, optimizer, scheduler, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    scaler = GradScaler()  # Mixed Precision Training용 스케일러
    
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, CONFIG.N_EPOCHS+1):
        model.train()
        train_loss = []
        for batch in tqdm(iter(train_loader)):
            features, lengths, labels = batch
            features = features.float().to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)  # Change for CrossEntropyLoss
            
            optimizer.zero_grad()
            
            with autocast():  # Mixed Precision Training
                output = model(features, lengths)
                loss = criterion(output, labels)
            
            scaler.scale(loss).backward()  # Mixed Precision Training
            scaler.step(optimizer)  # Mixed Precision Training
            scheduler.step()
            scaler.update()  # Mixed Precision Training
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}]')
            
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
    
    return best_model

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
        for batch in tqdm(iter(val_loader)):
            features, lengths, labels = batch
            features = features.float().to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)  # Change for CrossEntropyLoss
            
            probs = model(features, lengths)
            
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


def fold_train(train_dataset, CONFIG ,device) :

    criterion = nn.BCELoss().to(device)
    
    fold = KFold(n_splits = CONFIG.N_FOLD, shuffle = True, random_state = CONFIG.SEED)

    for f, (train_idx, val_idx) in enumerate(fold.split(train_dataset)) : 
        print(f"========== FOLD {f} Train ==========")
        
        best_val_score = 0

        model = ResNetTransformerModel(input_channels=1, num_classes=CONFIG.N_CLASSES)
        model.to(device)


        optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.00001)
        warmup_ratio = 0.1
        t_total = len(train_dataset) * CONFIG.N_EPOCHS
        warmup_step = int(t_total * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_step, num_training_steps = t_total)
    
        
        train_ds = Subset(train_dataset, train_idx)
        val_ds = Subset(train_dataset, val_idx)

        train_dl = DataLoader(train_ds, batch_size = CONFIG.BATCH_SIZE , shuffle = True, num_workers = 4, collate_fn = collate_fn)
        val_dl = DataLoader(val_ds, batch_size = CONFIG.BATCH_SIZE, shuffle = False, num_workers = 4, collate_fn = collate_fn)

        best_m = train(model, optimizer, scheduler, train_dl, val_dl, device)

        torch.save(best_m.state_dict(), f'./fold_{f}_best.pth')