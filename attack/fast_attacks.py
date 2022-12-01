import numpy as np
from yolox_model import YoloxModel
from torch.autograd import Variable
import torch

def fgsm(model:YoloxModel,img:np.ndarray, eps:float, cuda=False):

    imgs = torch.from_numpy(img[None, :, :, :])
    if cuda:
        imgs = imgs.cuda()
    imgs = Variable(imgs, requires_grad=True)

    loss = model(imgs)
    loss.backward()
    grad = imgs.grad
    with torch.no_grad():
        perturbed_imgs = (imgs + (grad.sign() * eps))
        perturbed_imgs = torch.clip(perturbed_imgs, 0, 255)
    if cuda:
        return perturbed_imgs[0].cpu().numpy()
    else:
        return perturbed_imgs[0].numpy()

def it_fgsm(model:YoloxModel, img:np.ndarray, eps:int=4, cuda=False):
    imgs = np.asarray([img])
    imgs = torch.from_numpy(imgs)
    if cuda:
        imgs = imgs.cuda()
    imgs = Variable(imgs, requires_grad=True)
    iter = int(min(eps + 4, 1.25 * eps))

    for _ in range(iter):
        loss = model(imgs)
        loss.backward()
        grad = imgs.grad
        with torch.no_grad():
            imgs = (imgs + (grad.sign()))
            imgs = torch.clip(imgs, 0, 255)
        imgs = Variable(imgs, requires_grad=True)
    if cuda:
        return imgs[0].detach().cpu().numpy()
    else:
        return imgs[0].detach().numpy()