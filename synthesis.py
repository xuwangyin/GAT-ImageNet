import os
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
from pgd_attack import perturb
import numpy as np
from sklearn.metrics import roc_curve, auc as roc_auc
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--eps', type=int, choices=[40, 100], default=100)
parser.add_argument('--nrows', type=int, default=1, required=False)
parser.add_argument('checkpoint', default='checkpoints-norm_Linf-eps_0.3-step_size_0.001-steps_400/resnet_50_dog_epoch1_iter1000.pth')
args = parser.parse_args()


torch.manual_seed(0)
np.random.seed(0)


# Adapted from 
# https://github.com/MadryLab/robustness_applications/blob/master/generation.ipynb
DATA_SHAPE = 224
GRAIN = 4


def downsample(x, step=GRAIN):
    down = torch.zeros([len(x), 3, DATA_SHAPE//step, DATA_SHAPE//step])

    for i in range(0, DATA_SHAPE, step):
        for j in range(0, DATA_SHAPE, step):
            v = x[:, :, i:i+step, j:j+step].mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            ii, jj = i // step, j // step
            down[:, :, ii:ii+1, jj:jj+1] = v
    return down


def upsample(x, step=GRAIN):
    up = torch.zeros([len(x), 3, DATA_SHAPE, DATA_SHAPE])

    for i in range(0, DATA_SHAPE, step):
        for j in range(0, DATA_SHAPE, step):
            ii, jj = i // step, j // step
            up[:, :, i:i+step, j:j+step] = x[:, :, ii:ii+1, jj:jj+1]
    return up


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dog_dist_file = 'imagenet_dog_dist.pkl'
if os.path.exists(dog_dist_file):
  with open(dog_dist_file, 'rb') as f:
    dist = pickle.load(f)
else:
  rootdir = './data/RestrictedImageNet-dog-vs-others-symlink'
  traindir = os.path.join(rootdir, 'train')
  train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor()
    ]))
  
  train_sampler = None
  train_loader = torch.utils.data.DataLoader(
  train_dataset, batch_size=512, shuffle=(train_sampler is None),
  num_workers=8, pin_memory=False, sampler=train_sampler)
  target_images = []
  for i, (images, labels) in enumerate(train_loader):
    print('{}/{}'.format((i + 1) * train_loader.batch_size, len(train_loader.dataset)))
    target_images.append(images[labels == 1])
  
  target_images = torch.cat(target_images)
  down_flat = downsample(target_images).view(len(target_images), -1)
  mean = down_flat.mean(dim=0)
  down_flat = down_flat - mean.unsqueeze(dim=0)
  cov = down_flat.t() @ down_flat / len(target_images)
  dist = MultivariateNormal(mean, covariance_matrix=cov + 1e-4 * torch.eye(3 * DATA_SHAPE // GRAIN * DATA_SHAPE // GRAIN))
  with open(dog_dist_file, 'wb') as f:
    pickle.dump(dist, f)

num_samples = args.nrows*8
seeds = dist.sample((num_samples,)).view(num_samples, 3, DATA_SHAPE//GRAIN, DATA_SHAPE//GRAIN)
seeds.clamp_(0, 1)
seeds = upsample(seeds)

model = models.resnet50()
model.load_state_dict(torch.load(args.checkpoint))
model.to(device)
model.eval()

if args.eps == 40:
  attack_config = {'norm': 'L2', 'eps': 40, 'step_size': 1, 'steps': 60}
else:
  attack_config = {'norm': 'L2', 'eps': 100, 'step_size': 10, 'steps': 100}

print('attack: {}'.format(attack_config))

adv = perturb(model, seeds.to(device), random_start=False, **attack_config)

fig, axes = plt.subplots(nrows=args.nrows, ncols=8, figsize=(8, args.nrows))
for i, data, ax in zip(range(64), adv, axes.ravel()):
  img = data.cpu().numpy().transpose([1, 2, 0])
  ax.imshow(img)
  ax.set_axis_off()

for ax in axes.ravel():
  ax.xaxis.set_major_locator(plt.NullLocator())
  ax.yaxis.set_major_locator(plt.NullLocator())

plt.subplots_adjust(top = 1, bottom = 0, right=1, left=0, hspace = 0.05, wspace = 0.05)

plt.show()
