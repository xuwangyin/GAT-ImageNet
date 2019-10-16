import numpy as np
from sklearn.metrics import roc_curve, auc as roc_auc
import torch
from pgd_attack import forward, perturb

def set_bn_eval(module):
  for submodule in module.modules():
    if 'batchnorm' in submodule.__class__.__name__.lower():
      submodule.train(False)


def set_train(model):
  """Disable batch normalization when training."""
  model.train()
  set_bn_eval(model)


def set_eval(model):
  model.eval()


def eval_auc(model, data_loader, attack_config, device, adv=True):
  """
  Compute AUC on a dataset comprised of pos and neg samples.

  Parameters
  ----------
  adv: bool, optional.
    If true, perturb negative samples, otherwise
    use negative samples as is. Default to True.
  """
  assert not model.training
  logits = []
  labels = []
  for i, (batch_images, batch_labels) in enumerate(data_loader):
    batch_images = batch_images.to(device)
    if adv:
      target = batch_images[batch_labels == 1]
      others = batch_images[batch_labels == 0]
      if others.nelement() > 0:
        others_adv = perturb(model, others,
                             random_start=False, **attack_config)
        batch_images = torch.cat([target, others_adv], 0)
      else:
        batch_images = target
    with torch.no_grad():
      batch_logits = forward(model, batch_images)
    logits.append(batch_logits.cpu().numpy())
    labels.append(batch_labels.numpy())
  logits = np.concatenate(logits)
  labels = np.concatenate(labels)
  fpr_, tpr_, thresholds = roc_curve(labels, logits)
  return roc_auc(fpr_, tpr_)
