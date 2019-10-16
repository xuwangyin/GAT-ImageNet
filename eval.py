import os
import sys
import copy
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn as nn
from utils import set_bn_eval, set_train, set_eval, eval_auc

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=10, required=False)
parser.add_argument('--step_size', type=float, default=0.002, required=False)
parser.add_argument('--eps', type=float, default=0.02, required=False)
parser.add_argument('--norm', type=str, choices=['Linf', 'L2'], default='Linf')
parser.add_argument('checkpoint')
args = parser.parse_args()
print(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
rootdir = './data/RestrictedImageNet-dog-vs-others-symlink'
valdir = os.path.join(rootdir, 'val')

val_loader = torch.utils.data.DataLoader(
  datasets.ImageFolder(valdir, transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
  ])),
  batch_size=64, shuffle=False,
  num_workers=4, pin_memory=True)

model = models.resnet50()
model.load_state_dict(torch.load(args.checkpoint))
model.to(device)

attack_config = {'norm': args.norm, 
                 'eps': args.eps, 
                 'step_size': args.step_size, 
                 'steps': args.steps}
set_eval(model)
val_nat_auc = eval_auc(model, val_loader, attack_config, device, adv=False)
val_adv_auc = eval_auc(model, val_loader, attack_config, device, adv=True)
print('val nat auc {}, adv auc {}'.format(val_nat_auc, val_adv_auc))
