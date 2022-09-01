'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from .get_plt_model import ModelWrapper


def train(
        model, 
        train_loader, val_loader, 
        learning_rate, trainer_args, save_path, optimizer):
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        filename="best",
        dirpath=save_path,
        save_last=True,
    )
    trainer_args['callbacks'] = [checkpoint_callback]
    plt_model = ModelWrapper(
        model, 
        learning_rate=learning_rate, 
        epochs=trainer_args['max_epochs'],
        optimizer=optimizer)
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(plt_model, train_loader, val_loader)
