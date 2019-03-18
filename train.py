import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import argparse

import program

ap = argparse.ArgumentParser(description='train.py')
# Command Line ardguments

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
#ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--dir', dest="dir", action="store", default="./classifier.pth")
ap.add_argument('--lr', dest="lr", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--structure', dest="structure", action="store", default="alexnet", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
pa = ap.parse_args()
where = pa.data_dir
path = pa.dir
lr = pa.lr
#structure = pa.arch
dropout = pa.dropout
hidden_units = pa.hidden_units
#power = pa.gpu
epochs = pa.epochs
image_datasets, validation_datasets, test_datasets, dataloaders, vloaders, tloaders = program.load_data()
model, optimizer, criterion = program.nn_build(lr,hidden_units,dropout)
program.train_nn(dataloaders, vloaders, model, optimizer, criterion, epochs, print_every=20)
program.save_checkpoint(image_datasets,path,hidden_units,dropout,lr,epochs)
print(" Training of the model is done")