import os
import sys
from pathlib import Path
import shutil
import torch
import torchvision.datasets as datasets

# Change this to your ImageNet directory
imagenet_dir = sys.argv[1]
data_dir = './data/RestrictedImageNet-dog-vs-others-symlink/'


intervals = {'dog': [151, 268], 'cat': [281, 285], 'frog': [30, 32],
          'turtle': [33, 37], 'bird': [80, 100], 'primate': [365, 382],
          'fish': [389, 397], 'crab': [118, 121], 'insect': [300, 319]}
dog_begin, dog_end = intervals['dog']

filled_range = []
for k, (left, right) in intervals.items():
  filled_range.extend(range(left, right+1))

# Create dirs 
for split in ['train', 'val']:
  for classidx in filled_range:
    if dog_begin <= classidx <= dog_end:
      path = os.path.join(data_dir, split, '1', str(classidx))
    else:
      path = os.path.join(data_dir, split, '0', str(classidx))
    Path(path).mkdir(parents=True, exist_ok=True)

for split in ['train', 'val']:
  dataset = datasets.ImageNet(root=imagenet_dir, split=split)
  for img, classidx in dataset.imgs:
    if classidx in filled_range:
      if dog_begin <= classidx <= dog_end:
        dst = os.path.join(data_dir, split, '1', str(classidx))
      else:
        dst = os.path.join(data_dir, split, '0', str(classidx))
      os.symlink(img, os.path.join(dst, os.path.basename(img)))
      # shutil.copy(img, dst) # or copy 
      print('linked {} {} -> {}'.format(classidx, img, dst))

