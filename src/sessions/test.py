import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import pytorch_lightning as pl
from .get_plt_model import ModelWrapper

def get_checkpoint_file(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            return file

def plt_model_load(model, checkpoint):
    state_dict = torch.load(checkpoint)['state_dict']
    model.load_state_dict(state_dict)
    return model

def test(model, test_loader, trainer_args, load_path):
    plt_model = ModelWrapper(model)
    if load_path is not None:
        if load_path.endswith(".ckpt"):
            checkpoint = load_path
        else:
            if load_path.endswith("/"):
                checkpoint = load_path + "best.ckpt"
            else:
                raise ValueError("if it is a directory, if must end with /; if it is a file, it must end with .ckpt")
        plt_model = plt_model_load(plt_model, checkpoint)
        print(f"Loaded model from {checkpoint}")

    trainer = pl.Trainer(**trainer_args)
    trainer.test(plt_model, test_loader)
