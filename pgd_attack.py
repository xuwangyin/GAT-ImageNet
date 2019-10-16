from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from attack_steps import L2Step, LinfStep


def forward(model, x):
  mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)
  std = torch.as_tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)
  mean = mean[None, :, None, None]
  std = std[None, :, None, None]
  # return model((x - mean)/std)[:, 151:269].mean(dim=1)
  return model((x - mean)/std)[:, 151]


def __clip_normalize_forward(model, x):
  x = torch.clamp(x, 0, 1)
  return forward(model, x)


def perturb(model, x, norm, eps, step_size, steps, random_start):
  """Perform PGD attack."""
  assert not model.training
  assert not x.requires_grad

  x0 = x.clone().detach()
  step_class = L2Step if norm == 'L2' else LinfStep
  step = step_class(eps=eps, orig_input=x0, step_size=step_size)

  if random_start:
    x = step.random_perturb(x)

  for i in range(steps):
    x = x.clone().detach().requires_grad_(True)
    logits = forward(model, x)
    loss = logits.mean()
    grad, = torch.autograd.grad(loss, [x])
    with torch.no_grad():
      x = step.step(x, grad)
      x = step.project(x)
  return x.clone().detach()
