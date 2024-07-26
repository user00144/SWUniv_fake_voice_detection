from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
import random
import librosa
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2FeatureExtractor


class CustomDataset(Dataset) :
    def __init__(self, df, CONFIG, is_train) :
        self.df = df
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("HyperMoon/wav2vec2-base-finetuned-deepfake-0919")
        self.max_len = CONFIG.EMB_LEN
        self.is_train = is_train
        self.CONFIG = CONFIG

    def __getitem__(self, idx) :
        path = self.df.iloc[idx].path
        if self.is_train :
            label = self.df.iloc[idx].label
            label_vector = np.zeros(2, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            data = self.load_data(path).squeeze()
            return data, label_vector
        return self.load_data(path).squeeze()

    def load_data(self, path) :
        y, sr = librosa.load(os.path.join(self.CONFIG.ROOT_FOLDER, path ), sr=self.CONFIG.SR)
        y = librosa.resample(y, orig_sr = sr, target_sr = self.CONFIG.RESAMPLE_SR)
        feat = self.feature_extractor(y, sampling_rate=self.CONFIG.RESAMPLE_SR, max_length=self.max_len, truncation=True, padding='max_length', return_tensors="pt")
        return self.min_max(feat.input_values)

    def min_max(self, ts) :
        return ((ts - torch.min(ts)) / (torch.max(ts) - torch.min(ts)))
        
    def __len__(self) :
        return len(self.df)


def get_dataloader(CONFIG, train_df, test_df, val = True) :
    if val :
        train, val = train_test_split(train_df, test_size=0.2, random_state=CONFIG.SEED)
    
        train_dataset = CustomDataset(train, CONFIG ,is_train = True)
        val_dataset = CustomDataset(val, CONFIG, is_train = True)
        test_dataset = CustomDataset(test_df, CONFIG, is_train = False)
    
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=CONFIG.BATCH_SIZE, num_workers = 4)
        val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=CONFIG.BATCH_SIZE, num_workers = 4)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=CONFIG.BATCH_SIZE, num_workers = 4)


        return train_dataloader, val_dataloader, test_dataloader

    else :
        train_dataset = CustomDataset(train_df, CONFIG, is_train = True)
        return train_dataset

