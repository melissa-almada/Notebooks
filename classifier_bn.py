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
from fastai.callback.wandb import WandbCallback

import warnings
warnings.filterwarnings('ignore')

import argparse
import wandb

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

def get_model(model_name='resnet34'):
    models = {
        'resnet34': resnet34,
        'resnet50': resnet50,
        'efficientnet_b0': efficientnet_b0,
        'densenet121': densenet121,
        'convnext_tiny': convnext_tiny
    }
    return models.get(model_name)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', type=str, default='resnet34')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--cross-validate', action='store_true')
    return parser.parse_args()

def cross_validate(args, n_folds=5):
    accuracies = []
    for fold in range(n_folds):
        # Create DataLoaders with different random seed for each fold
        dls = DataLoaders.from_dataset(GWDataset(SIG_PATH, BG_PATH), 
                                     splitter=RandomThreeSplitter(seed=fold),
                                     batch_size=args.batch_size)
        
        learn = cnn_learner(dls, get_model(args.model))

def plot_results(learn):
    """Plot training metrics"""
    learn.recorder.plot_loss()
    plt.savefig(f'{LOG_PATH}/loss.png')
    plt.close()

def main():
    args = parse_args()
    
    # Initialize wandb
    wandb.init(project='gw-classification')
    
    # Create DataLoaders
    dls = DataLoaders.from_dataset(GWDataset(SIG_PATH, BG_PATH),
                                 splitter=RandomThreeSplitter(),
                                 batch_size=args.batch_size)
    
    # Enhanced transforms
    transforms = [
        Resize(112),
        Rotate(degrees=(-10,10)),
        Brightness(change=(0.4,0.6))
    ]
    
    # Enhanced metrics
    metrics = [accuracy, Precision(), Recall(), F1Score()]
    
    # Enhanced callbacks
    callbacks = [
        ShowGraphCallback(),
        SaveModelCallback(monitor='accuracy'),
        EarlyStoppingCallback(monitor='accuracy', patience=3),
        WandbCallback()
    ]
    
    if args.cross_validate:
        acc_mean, acc_std = cross_validate(args, n_folds=5)
        print(f'Cross-validation accuracy: {acc_mean:.3f} ± {acc_std:.3f}')
    else:
        learn = cnn_learner(dls, 
                          get_model(args.model), 
                          metrics=metrics,
                          cbs=callbacks)
        learn.fit_one_cycle(args.epochs)
        plot_results(learn)

if __name__ == "__main__":
    main()