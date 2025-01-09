# Imports
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os
from pathlib import Path

from sklearn import metrics
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as tfms
from torchvision import models

from fastai.basics import *
from fastai.vision.all import *

import warnings
warnings.filterwarnings('ignore')

# Paths and variables
OUTPUT_PATH = './outputs/'
SIG_PATH = OUTPUT_PATH + 'sig/'
BG_PATH = OUTPUT_PATH + 'bg/'
CONFIG_PATH = OUTPUT_PATH + 'config/'
LOG_PATH = './model_logs/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

class GWDataset(Dataset):
    """Dataset for Gravitational Wave data"""
    def __init__(self, sig_path, bg_path):
        self.sig_files = [f for f in Path(sig_path).glob('*.npz')]
        self.bg_files = [f for f in Path(bg_path).glob('*.npz')]
        self.all_files = self.sig_files + self.bg_files
        self.labels = [1]*len(self.sig_files) + [0]*len(self.bg_files)
        
    def __len__(self):
        return len(self.all_files)
        
    def __getitem__(self, idx):
        # Load npz file
        data = np.load(self.all_files[idx])
        qgraph = data['qgraph']  # Shape: [3, height, width]
        
        # Convert to tensor and normalize
        tensor = torch.FloatTensor(qgraph)
        tensor = (tensor - tensor.mean()) / tensor.std()
        
        return tensor, self.labels[idx]

def RandomThreeSplitter(valid_pct=0.15, test_pct=0.15, seed=None):
    def _inner(o):
        if seed is not None: torch.manual_seed(seed)
        rand_idx = L(list(torch.randperm(len(o)).numpy()))
        cut_val = int(valid_pct * len(o))
        cut_test = cut_val + int(valid_pct * len(o))
        return rand_idx[cut_test:],rand_idx[:cut_val], rand_idx[cut_val:cut_test]
    return _inner

def main():
    # Create dataset
    dataset = GWDataset(SIG_PATH, BG_PATH)
    
    # Create DataBlock
    transforms = [Resize(112)]  # No need for ToTensor as data is already in tensor format
    gw_block = DataBlock(
        blocks=(TransformBlock, CategoryBlock),
        get_items=lambda _: range(len(dataset)),
        get_x=lambda i: dataset[i][0],
        get_y=lambda i: dataset[i][1],
        splitter=RandomThreeSplitter(valid_pct=0.15, test_pct=0.15, seed=42),
        item_tfms=transforms
    )

    # Create DataLoaders
    dls = gw_block.dataloaders(
        dataset,
        bs=64,
        device=device
    )

    # Create and train model
    learn = cnn_learner(
        dls,
        resnet34,
        pretrained=False,
        path=LOG_PATH,
        metrics=[accuracy],
        cbs=[ShowGraphCallback()]
    )

    # Find learning rate
    get_lr = learn.lr_find(suggest_funcs=(steep, valley, minimum))
    print(f'Minimum/10: {get_lr.minimum:.2e}, steepest point: {get_lr.steep:.3e}')

    # Train model
    learn.fit_one_cycle(10, lr_max=get_lr.valley)

    # Interpret results
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    plt.show()

    interp.plot_top_losses(5, nrows=1)
    plt.show()
    
    # Save the model
    learn.export('gw_classifier.pkl')

if __name__ == "__main__":
    main()