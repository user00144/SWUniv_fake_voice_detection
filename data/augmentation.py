import pandas as pd
import numpy as np
import librosa
import torchaudio
import os
import torch
from tqdm import tqdm
import random

def generate_white_noise(length):
    return torch.randn(length)

def generate_pink_noise(length):
    num_columns = 16
    array = torch.randn(length, num_columns)
    array = torch.cumsum(array, dim=0)
    array -= torch.mean(array, dim=1, keepdim=True)
    return array[:, -1]

def generate_brown_noise(length):
    return torch.cumsum(torch.randn(length), dim=0)

def add_noise_to_waveform(waveform, noise_waveform, noise_level):
    # Adjust the length of the noise to match the waveform length
    noise_waveform = noise_waveform[:waveform.shape[1]]
    noise_waveform = noise_waveform.unsqueeze(0)
    # Add noise to the original waveform
    augmented_waveform = waveform + noise_level * noise_waveform
    # Normalize to the original signal's amplitude range
    augmented_waveform = augmented_waveform / torch.max(torch.abs(augmented_waveform))
    return augmented_waveform


def augment_with_random_noise(waveform, sample_rate, noise_level=0.2):
    length = waveform.shape[1]
    
    # Randomly choose noise type
    noise_types = ['white', 'pink', 'brown']
    noise_type = random.choice(noise_types)
    
    if noise_type == 'white':
        noise_waveform = generate_white_noise(length)
    elif noise_type == 'pink':
        noise_waveform = generate_pink_noise(length)
    else :
        noise_waveform = generate_brown_noise(length)
    
    return add_noise_to_waveform(waveform, noise_waveform, noise_level)


df_train = pd.read_csv("train.csv")

df_aug = df_train.sample(frac = 1, random_state = 21)

savepath = "./aug"

df_list = []

for i in tqdm(range(len(df_aug))) :
    df_path = df_aug.iloc[i].path
    df_id = df_aug.iloc[i].id
    df_label = df_aug.iloc[i].label
    df_save_path = os.path.join(savepath, f"aug_{df_id}.wav")
    waveform, sr = torchaudio.load(os.path.join("./", df_path))
    waveform = augment_with_random_noise(waveform, sr)
    torchaudio.save(df_save_path, waveform, sample_rate = sr)
    df_list.append([df_id, df_save_path, df_label])

df__aug = pd.DataFrame(data = df_list, columns = ['id', 'path', 'label'])

df__aug.to_csv('aug.csv', index = False)





