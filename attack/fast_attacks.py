import numpy as np
from torch.autograd import Variable
from torch import Tensor
import torch.nn as nn
import torch


def fgsm(model:nn.Module,imgs:Tensor, device: torch.device, eps:int=4, return_numpy=False) -> None:

    imgs = imgs.to(device)
    imgs.requires_grad = True
    # imgs = Variable(imgs, requires_grad=True)

    ## this will stop the calculation for the graident with respect to input which is faster 
    for param in model.parameters():
        param.requires_grad = False

    model.train()
    loss = model(imgs)
    loss.backward()
    grad = imgs.grad
    with torch.no_grad():
        perturbed_imgs = (imgs + (grad.sign() * eps))
        perturbed_imgs = torch.clip(perturbed_imgs, 0, 255)

    if return_numpy:
        return perturbed_imgs.detach().cpu().numpy()
    else:
        return perturbed_imgs


def it_fgsm(model:nn.Module, imgs:Tensor, device: torch.device,eps:int=4, return_numpy=False):
    imgs = imgs.to(device)
    imgs.requires_grad = True

    iter = int(min(eps + 4, 1.25 * eps))

    model.train()
    for _ in range(iter):
        loss = model(imgs)
        loss.backward()
        grad = imgs.grad
        with torch.no_grad():
            imgs = (imgs + (grad.sign()))
            imgs = torch.clip(imgs, 0, 255)
        imgs.requires_grad = True
    if return_numpy:
        return imgs.detach().cpu().numpy()
    else:
        return imgs