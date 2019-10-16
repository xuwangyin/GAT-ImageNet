import os
import sys
import copy
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as roc_auc
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
from pgd_attack import forward, perturb
from utils import set_bn_eval, set_train, set_eval, eval_auc


parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=40, required=False)
parser.add_argument('--step_size', type=float, default=0.001, required=False)
parser.add_argument('--eps', type=float, default=0.02, required=False)
parser.add_argument('--norm', type=str, choices=['Linf', 'L2'], default='Linf')
args = parser.parse_args()
print(args)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rootdir = './data/RestrictedImageNet-dog-vs-others-symlink'
traindir = os.path.join(rootdir, 'train')
valdir = os.path.join(rootdir, 'val')

train_dataset = datasets.ImageFolder(
  traindir,
  transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
  ]))

train_sampler = None
train_loader = torch.utils.data.DataLoader(
  train_dataset, batch_size=64, shuffle=(train_sampler is None),
  num_workers=4, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
  datasets.ImageFolder(valdir, transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
  ])),
  batch_size=64, shuffle=False,
  num_workers=4, pin_memory=True)

model = models.resnet50(pretrained=True)
# Linearly scale the perturbation limit, start from 0.005
#model.load_state_dict(torch.load('checkpoints-norm_Linf-eps_0.005-step_size_0.001-steps_10/resnet_50_dog_epoch3_iter4000.pth'))
model.load_state_dict(torch.load('checkpoints-norm_Linf-eps_0.01-step_size_0.001-steps_20/resnet_50_dog_epoch3_iter4000.pth'))
model.to(device)

attack_config = {'norm': args.norm, 
                 'eps': args.eps, 
                 'step_size': args.step_size, 
                 'steps': args.steps}

# Record nat auc and adv auc before training
set_eval(model)
val_nat_auc = eval_auc(model, val_loader, attack_config, device, adv=False)
val_adv_auc = eval_auc(model, val_loader, attack_config, device, adv=True)
print('epoch {}, val nat auc {}, adv auc {}'.format(0, val_nat_auc, val_adv_auc))

criterion = nn.BCEWithLogitsLoss()
# Follow https://github.com/pytorch/examples/blob/master/imagenet/main.py
optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4)

ckpt_dir = 'checkpoints-' + '-'.join(['{}_{}'.format(k, v) for k, v in attack_config.items()])
Path(ckpt_dir).mkdir(exist_ok=True, parents=True)

for epoch in range(1, 100):
  # train
  set_train(model)
  for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    target = images[labels == 1]
    others = images[labels == 0]
    if others.nelement() == 0:
      continue
    set_eval(model)
    others_adv = perturb(model, others, random_start=True, **attack_config)
    set_train(model)
    images = torch.cat([target, others_adv], 0)
    labels = torch.cat([torch.ones(target.shape[0]),
                        torch.zeros(others_adv.shape[0])]).type(labels.dtype).to(device)

    optimizer.zero_grad()
    logits = forward(model, images)
    loss = criterion(logits, labels.type(images.dtype))
    loss.backward()
    optimizer.step()

    fpr_, tpr_, thresholds = roc_curve(labels.data.cpu().numpy(), logits.data.cpu().numpy())
    auc = roc_auc(fpr_, tpr_)
    print('epoch {} iter {}/{} ({}) loss {} auc {}'.format(epoch, i, len(train_loader.dataset)//train_loader.batch_size, datetime.now(), loss.item(), auc))

    if (i+1) % 1000 == 0:
      torch.save(model.state_dict(), '{}/resnet_50_dog_epoch{}_iter{}.pth'.format(ckpt_dir, epoch, i+1))

      # eval
      set_eval(model)
      val_nat_auc = eval_auc(model, val_loader, attack_config, device, adv=False)
      val_adv_auc = eval_auc(model, val_loader, attack_config, device, adv=True)
      print('epoch {}, iter {}, val nat auc {}, adv auc {}'.format(epoch, i+1, val_nat_auc, val_adv_auc))

