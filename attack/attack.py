import cv2
import numpy as np
from yolox_model import YoloxModel
from torch.autograd import Variable
import torch
from helper_functions import preprocess

def fgsm(model:YoloxModel,original_image:np.ndarray, eps:float):
    preprocessed_img = preprocess(original_image)
    imgs = np.asarray([preprocessed_img])
    imgs = Variable(torch.from_numpy(imgs), requires_grad=True)

    loss = model(imgs)
    loss.backward()
    grad = imgs.grad
    perturbed_imgs = (imgs + (grad.sign() * eps)).detach()
    perturbed_imgs = torch.clip(perturbed_imgs, 0, 255)
    return perturbed_imgs[0]

def it_fgsm(model:YoloxModel, original_image:np.ndarray, eps:int=4):
    preprocessed_img = preprocess(original_image)
    imgs = np.asarray([preprocessed_img])
    imgs = Variable(torch.from_numpy(imgs), requires_grad=True)
    iter = int(min(eps + 4, 1.25 * eps))
    for _ in range(iter):
        loss = model(imgs)
        loss.backward()
        grad = imgs.grad
        imgs = (imgs + (grad.sign())).detach()
        imgs = torch.clip(imgs, 0, 255)
        imgs = Variable(imgs, requires_grad=True)
    return imgs[0].detach()