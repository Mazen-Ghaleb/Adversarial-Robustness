from attack.attack_base import AttackBase
import torch
from torch import Tensor
from torch.nn import Module
from typing import Callable

class ItFGSM(AttackBase):
    def __init__(self) -> None:
        super(ItFGSM, self).__init__()

    def generate_attack(self,images:Tensor, targets=None,eps:int=4, return_numpy=False):
        super().generate_attack(images, targets, eps, return_numpy)
        images.requires_grad = True

        iter = int(min(eps + 4, 1.25 * eps))

        self.model.train()
        for _ in range(iter):
            modle_outputs = self.model(images)
            if targets is None:
                targets = self.target_generator(modle_outputs)
            loss = self.loss(modle_outputs, targets)
            loss.backward()
            grad = images.grad
            with torch.no_grad():
                images = (images + (grad.sign()))
                images = torch.clip(images, 0, 255)
            images.requires_grad = True
            
        if return_numpy:
            return images.detach().cpu().numpy()
        else:
            return images