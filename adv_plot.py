import os
import sys
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data
import torch.nn as nn
from pgd_attack import perturb
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet50()
model.load_state_dict(torch.load('checkpoints-norm_Linf-eps_0.3-step_size_0.001-steps_400/resnet_50_dog_epoch1_iter1000.pth'))
model.to(device)

model.eval()

# # Save images with top activations
# num_sampels = 100
# val_images = []
# for i, (batch_images, batch_labels) in enumerate(val_loader):
#   val_images.append(batch_images)
# val_images = np.concatenate(val_images)
# adv_logits = np.load('data/adv_logits.npy')
# indices = np.argsort(adv_logits)[::-1][:num_sampels]
# np.save('data/top_images100.npy', val_images[indices])

# Dog face retouching
top_images100 = np.load('data/top_images100.npy')
dog_indices = [7, 8, 11, 18, 26, 39, 40, 41, 42, 47, 52, 53, 55, 65, 69, 75, 80, 96]
dog_images = top_images100[dog_indices[:9]]
num_dog_sampels = dog_images.shape[0]
attack_config = {'norm': 'L2', 'eps': 30, 'step_size': 5, 'steps': 100}
dog_images_adv = perturb(model, torch.from_numpy(dog_images).to(device), random_start=False, **attack_config)
dog_samples = np.concatenate([dog_images, dog_images_adv.cpu().numpy()])

fig, axes = plt.subplots(nrows=2, ncols=num_dog_sampels, figsize=(num_dog_sampels, 2))
for i, data, ax in zip(range(100), dog_samples, axes.ravel()):
  img = data.transpose([1, 2, 0])
  ax.imshow(img)
  ax.set_axis_off()

for ax in axes.ravel():
  ax.xaxis.set_major_locator(plt.NullLocator())
  ax.yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top = 1, bottom = 0, right=1, left=0, hspace = 0.05, wspace = 0.05)
#plt.savefig('imagenet_dog_enhance_part1.pdf', dpi=200)

# Adversarial examples
others_indices = [0, 29, 35, 37, 43, 48, 49, 51, 54, 57, 62, 66, 70, 73, 76, 86, 88, 89]
others_images = top_images100[others_indices[:9]]
num_others_sampels = others_images.shape[0]
attack_config = {'norm': 'L2', 'eps': 30, 'step_size': 5, 'steps': 100}
others_images_adv = perturb(model, torch.from_numpy(others_images).to(device), random_start=False, **attack_config)
others_samples = np.concatenate([others_images, others_images_adv.cpu().numpy()])

fig, axes = plt.subplots(nrows=2, ncols=num_others_sampels, figsize=(num_others_sampels, 2))
for i, data, ax in zip(range(100), others_samples, axes.ravel()):
  img = data.transpose([1, 2, 0])
  ax.imshow(img)
  ax.set_axis_off()

for ax in axes.ravel():
  ax.xaxis.set_major_locator(plt.NullLocator())
  ax.yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top = 1, bottom = 0, right=1, left=0, hspace = 0.05, wspace = 0.05)
# plt.savefig('imagenet_adv_attack_part2.pdf', dpi=200)
plt.show()

