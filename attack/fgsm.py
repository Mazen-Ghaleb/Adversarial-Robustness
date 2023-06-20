from attack.attack_base import AttackBase
from torch.nn import Module
import torch
from typing import Callable

class FGSM(AttackBase):
    def __init__(self, target_generator, loss) -> None:
        super(FGSM, self).__init__(target_generator, loss)

    def generate_attack(self, images, targets=None, eps=4, return_numpy=False):
        super().generate_attack(images, targets, eps, return_numpy)
        images.requires_grad = True

        self.model.eval()
        model_output = self.model(images)
        if targets is None:
            targets = self.target_generator(model_output)

        loss = self.loss(model_output, targets)
        loss.backward()
        grad = images.grad

        with torch.no_grad():
            perturbed_imgs = (images + (grad.sign() * eps))
            perturbed_imgs = torch.clip(perturbed_imgs, 0, 255)

        if return_numpy:
            return perturbed_imgs.detach().cpu().numpy()
        else:
            return perturbed_imgs
