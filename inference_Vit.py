import torch
import numpy as np
import pandas as pd
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F


from conf.configure import Config

CONFIG = Config()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

test_df = pd.read_csv(os.path.join(CONFIG.ROOT_FOLDER, 'test.csv'))

from dataset.spectrogram_dataset import AudioDataset

test_dataset = AudioDataset(test_df, False, CONFIG)

test_loader = DataLoader(test_dataset, shuffle=False, batch_size=CONFIG.BATCH_SIZE, num_workers = 4)


print("========== Test Dataset Inference ==========")
all_preds = []
temp = np.zeros((50000, 2))

from models.ViT import ViTmodel

model = ViTmodel()


with torch.no_grad():
    for fold in range(CONFIG.N_FOLD) :
        model.load_state_dict(torch.load(f"fold_{fold}_best.pth"))
        model.to(device)
        model.eval()
        predictions = []
        for features in tqdm(iter(test_loader)):
            features = features.float().to(device)
            probs = model(features)
            probs  = probs.cpu().detach().numpy()
            predictions += probs.tolist()
        predictions = np.array(predictions)
        all_preds.append(predictions)

for preds in all_preds :
    temp += preds
temp /= CONFIG.N_FOLD

submit = pd.read_csv(os.path.join(CONFIG.ROOT_FOLDER,'sample_submission.csv'))
submit.iloc[:, 1:] = temp
print(submit.head())
submit.to_csv('./submit.csv', index=False)
