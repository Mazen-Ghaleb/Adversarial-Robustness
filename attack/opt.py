import torch
from attack_base import AttackBase
from torch.optim import Adam
class OPT(AttackBase):
    def __init__(self, optimizer = None) -> None:
        super(OPT, self).__init__()
        if optimizer is None:
            self.optimzer = Adam(lr=0.01)
        else:
            self.optimzer = optimizer
    
    def generate_attack(self, images, targets=None, return_numpy=False):
        with torch.no_grad():
            model_output = self.model(images)
            targets = self.target_generator(model_output)
            eps = torch.randint(0, 10, images.shape)
            perturbed_image = images + eps
        
        

        for i in range(10):
            model_out = self.model(images + eps)
            loss = torch.norm()

