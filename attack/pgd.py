from attack.attac_base import AttackBase
import torch
class PGD(AttackBase):
    def __init__(self) -> None:
        super().__init__()

    def __pgd_step(self, images, targets = None, eps=4, step_norm='inf', eps_norm='inf'):

        images.requires_grad = True
        model_output = self.model(images)
        
        loss = self.loss(model_output, targets)
        loss.backward()

        with torch.no_grad():
            grad = images.grad
            if step_norm == 'inf':
                norm = grad.sign() * eps
            else:
                norm = grad.view(images.shape[0], -1).norm(p=step_norm, dim=-1)
                norm = norm.view(-1, images.shape[0], 1, 1)
                grad = grad * eps / norm

            perturbed_images = grad + images

            if eps_norm == 'inf':
                perturbed_images = torch.min(perturbed_images, images + eps)
                perturbed_images = torch.max(images + eps, perturbed_images)
            else:
                delta = perturbed_images - images
                delta_norm = delta.view(delta.shape[0], -1).norm(p=step_norm, dim=-1)
                mask = delta_norm <= eps
                delta_norm[mask] = delta_norm[mask]
                delta *= eps / delta_norm.view(-1, 1, 1, 1)

                perturbed_images = images + delta

            return torch.clip(perturbed_images, 0, 255)


        

    def generate_attack(self, images, targets=None, eps=4, return_numpy=False):
        super().generate_attack(images, targets, eps, return_numpy)
        self.model.train()
        perturbed_images = images
        from tqdm import tqdm
        model_output = self.model(images)
        if targets is None:
            targets = self.target_generator(model_output)
        for i in tqdm(range(100)):
            perturbed_images = self.__pgd_step(perturbed_images, targets=targets)
        return perturbed_images


