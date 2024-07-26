import librosa
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import warnings
from conf.configure import Config

CONFIG = Config()

class CustomDataset(Dataset):
    def __init__(self, df, train_mode, CONFIG, device):
        self.df = df
        self.train_mode = train_mode
        self.CONFIG = CONFIG
        self.device = device

        n_fft = 1024
        hop_length = 512
        n_mels = 128
 
        self.mfcc_transform = T.MFCC(sample_rate = 32000,
                    n_mfcc = 64,
                    melkwargs = {
                        "n_fft" : n_fft,
                        "n_mels" : n_mels,
                        "hop_length" : hop_length,
                        "mel_scale" : "htk",
                    },)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.train_mode :
            mfcc, label = self.get_mfcc_feature([self.df.iloc[index].path, self.df.iloc[index].label], self.train_mode)

            return mfcc, label
        else :
            mfcc = self.get_mfcc_feature([self.df.iloc[index].path], self.train_mode)
            return mfcc
            
    def get_mfcc_feature(self, row, train_mode=True, max_len=310):
        file_path = row[0]

        # 오디오 파일 로드
        data, sr = torchaudio.load(os.path.join(self.CONFIG.ROOT_FOLDER, file_path))
        mf = self.mfcc_transform(data).squeeze()

        length = len(mf[1])
        
        if train_mode:
            label = row[1]
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            if label == 'both' :
                label_vector[0] = 1
    
        if length < max_len :
            pad_width = max_len - length
            mf = F.pad(mf, pad =(0, pad_width,0,0), mode = "constant", value = 0)
        elif length > max_len :
            mf = mf[:,:max_len]

        if train_mode:
            return mf, torch.tensor(label_vector)
        else :
            return mf

