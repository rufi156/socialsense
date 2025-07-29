import os
import torch
import torch.nn as nn
from tqdm.notebook import tqdm, trange
import torch.optim as optim
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import random
import pickle
import datetime
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# DATASET_DIR = (Path("..") / ".." / "datasets").resolve()
# DATASETS = ["OFFICE-MANNERSDB", "MANNERSDBPlus"]
# LABEL_COLS = [
#     "Vaccum Cleaning", "Mopping the Floor", "Carry Warm Food",
#     "Carry Cold Food", "Carry Drinks", "Carry Small Objects",
#     "Carry Large Objects", "Cleaning", "Starting a conversation"
# ]

import sys
sys.path.append('../..')

from data_processing.data_processing import create_dataloaders


def get_dataloader(df, batch_sizes=(32, 64, 64), resize_img_to=(224,224), return_splits=False, double_img=False, transforms=None, num_workers=0):
    """
    for yolo_depthanything model 224,224
    otherwise 512, 288
    LGR had 128,128 
    MobileNetv2 had 224, 224
    """

    # Create domain-specific dataloaders
    domains = df['domain'].unique()
    domain_dataloaders = {}
    test_split_idx = set()
    for domain in domains:
        domain_df = df[df['domain'] == domain]
        #domain_df = domain_df.sample(frac=0.5, random_state=42)
        loaders, split_idx = create_dataloaders(domain_df, batch_sizes=batch_sizes, resize_img_to=resize_img_to, seed=SEED, return_splits=True, double_img=double_img, transforms=transforms, num_workers=num_workers)
        domain_dataloaders[domain] = loaders
        test_split_idx.update(set(split_idx['test']))

    return domain_dataloaders if not return_splits else (domain_dataloaders, test_split_idx)
