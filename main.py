import librosa

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch
import os

import warnings
warnings.filterwarnings('ignore')


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"


print('cuda : ', torch.cuda.is_available())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()

from conf.configure import Config
CONFIG = Config()



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG.SEED) # Seed 고정

#load dataframe
train_df = pd.read_csv(os.path.join(CONFIG.ROOT_FOLDER, 'train.csv'))
#aug_df = pd.read_csv(os.path.join(CONFIG.ROOT_FOLDER, 'aug.csv'))
aug_both_df = pd.read_csv(os.path.join(CONFIG.ROOT_FOLDER, 'aug_both.csv'))

#aug_df = aug_df.sample(frac = 1, random_state = CONFIG.SEED)
aug_both_df = aug_both_df.sample(frac = 0.3, random_state = CONFIG.SEED)

train_df = pd.concat([train_df ,aug_both_df], axis = 0)
train_df = train_df.reset_index(drop = True)

#load AASIST Model
#from models.AASIST import Model
#model = Model(CONFIG.conf_dic["model_config"])

#load DataLoader
from dataset.mfcc_dataset import CustomDataset
train_dataset = CustomDataset(train_df, True ,CONFIG, device)

#train function
#from train_spec_fold import fold_train
from train_fold import train_fold
#configure optimizer & scheduler
train_fold(train_dataset, CONFIG)
