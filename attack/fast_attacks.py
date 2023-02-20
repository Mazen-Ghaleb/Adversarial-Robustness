import numpy as np
from torch.autograd import Variable
from torch import Tensor
import torch.nn as nn
import torch


def fgsm(model:nn.Module, model_input:Tensor, device: torch.device, eps:int=4, return_numpy=False, batch = True) -> None:

    if not batch:
        model_input = np.asarray(model_input[None, :, :, :])
        model_input = torch.from_numpy(model_input)

    model_input = model_input.to(device)
    model_input.requires_grad = True
    # imgs = Variable(imgs, requires_grad=True)

    ## this will stop the calculation for the graident with respect to input which is faster 
    for param in model.parameters():
        param.requires_grad = False

    model.train()
    loss = model(model_input)
    loss.backward()
    grad = model_input.grad
    with torch.no_grad():
        perturbed_imgs = (model_input + (grad.sign() * eps))
        perturbed_imgs = torch.clip(perturbed_imgs, 0, 255)

    if return_numpy:
        if not batch:
            return perturbed_imgs.detach().cpu().numpy()[0]
        return perturbed_imgs.detach().cpu().numpy()
    else:
        if not batch:
            return perturbed_imgs[0]
        return perturbed_imgs


def it_fgsm(model:nn.Module, model_input:Tensor, device: torch.device,eps:int=4, return_numpy=False, batch = True):
    
    if not batch:
        model_input = np.asarray(model_input[None, :, :, :])
        model_input = torch.from_numpy(model_input)
    
    model_input = model_input.to(device)
    model_input.requires_grad = True

    iter = int(min(eps + 4, 1.25 * eps))

    model.train()
    for _ in range(iter):
        loss = model(model_input)
        loss.backward()
        grad = model_input.grad
        with torch.no_grad():
            model_input = (model_input + (grad.sign()))
            model_input = torch.clip(model_input, 0, 255)
        model_input.requires_grad = True
        
    if return_numpy:
        if not batch:
            return model_input.detach().cpu().numpy()[0]
        return model_input.detach().cpu().numpy()
    else:
        if not batch:
            return model_input[0]
        return model_input