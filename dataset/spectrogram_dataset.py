import librosa.display
import numpy as np
import pandas as pd
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import warnings
import torchaudio.transforms as transforms
from conf.configure import Config
from PIL import Image

CONFIG = Config()

from speechbrain.inference.separation import SepformerSeparation

from transformers import ViTFeatureExtractor, ViTImageProcessor


class CustomDataset(Dataset):
    def __init__(self, df, train_mode, CONFIG):
        self.df = df
        self.train_mode = train_mode
        self.CONFIG = CONFIG

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.train_mode :
            spectrogram, label = self.get_spectrogram_feature([self.df.iloc[index].path, self.df.iloc[index].label], self.train_mode)
            
            return torch.tensor(spectrogram), label
        else :
            spectrogram = self.get_spectrogram_feature([self.df.iloc[index].path], self.train_mode)

            return torch.tensor(spectrogram)
            
    def get_spectrogram_feature(self, row, train_mode=True, n_mels=128, max_len=155):
        file_path = row[0]

        # 오디오 파일 로드
        data, sr = librosa.load(os.path.join(self.CONFIG.ROOT_FOLDER, file_path))
        n_fft = 2048
        win_length = 2048
        hop_length = 1024
        n_mels = 224
 
        D = np.abs(librosa.stft(data, n_fft=n_fft, win_length = win_length, hop_length=hop_length))
        mel = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
        length = len(mel[1])
        
        if train_mode:
            label = row[1]
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
    
        if length < max_len :
            pad_width = max_len - length
            mel = np.pad(mel, ((0, 0),(0, pad_width)), mode='constant')
        elif length > max_len :
            mel = mel[:,:max_len]

        if train_mode:
            return mel, torch.tensor(label_vector)
        else :
            return mel

def collate_fn(batch):
    features = [item[0] for item in batch]
    lengths = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    features = torch.stack(features)
    lengths = torch.tensor(lengths)
    labels = torch.stack(labels)

    return features, lengths, labels

def collate_test_fn(batch):
    features = [item[0] for item in batch]
    lengths = [item[1] for item in batch]

    features = torch.stack(features)
    lengths = torch.tensor(lengths)

    return features, lengths


import torch
from torch.utils.data import Dataset
import librosa
from transformers import ViTFeatureExtractor
from PIL import Image
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, dataframe, train_mode, CONFIG):
        self.dataframe = dataframe
        #self.feature_extractor = ViTImageProcessor.from_pretrained("MattyB95/VIT-VoxCelebSpoof-Mel_Spectrogram-Synthetic-Voice-Detection")
        self.CONFIG = CONFIG
        self.train_mode = train_mode

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 오디오 파일 경로와 라벨 추출
        file_path = self.dataframe.iloc[idx].path

        # 오디오 파일 로드
        data, sr = librosa.load(os.path.join(self.CONFIG.ROOT_FOLDER, file_path))
        n_fft = 2048
        win_length = 2048
        hop_length = 1024
        n_mels = 224
 
        D = np.abs(librosa.stft(data, n_fft=n_fft, win_length = win_length, hop_length=hop_length))
        mel = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
        # 스펙트로그램을 이미지로 변환
        image = self.spectrogram_to_image(mel)
        image = image.convert("RGB")
        
        # ViT 모델의 입력 형식으로 변환
        #inputs = self.feature_extractor(images=image, return_tensors="pt")

        if self.train_mode:
            label = self.dataframe.iloc[idx].label
            label_vector = np.zeros(self.CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            if label == 'both' :
                label_vector[0] = 1
        # 이미지 텐서와 라벨 반환
            return image, torch.tensor(label_vector) #inputs['pixel_values'].squeeze()

        return image #inputs['pixel_values'].squeeze()

    def spectrogram_to_image(self, spectrogram):
        spec_min = spectrogram.min()
        spec_max = spectrogram.max()
        spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)  # normalize to [0, 1]
        spectrogram = (spectrogram * 255).astype(np.uint8)  # scale to [0, 255]
        image = Image.fromarray(spectrogram)
        return image

def get_dataset(train_df, CONFIG, device) :
    train_dataset = CustomDataset(train_df, True, CONFIG)
    return train_dataset


