from yolox.models import YOLOX as BaseYOLOX
import torch.nn.functional as F
import torch
class CustomYOLOX(BaseYOLOX):
    def __init__(self, backbone=None, head=None):
        super().__init__(backbone, head)

    def forward(self, x, targets=None):
        fpn_outs = self.backbone(x)
        return self.head(fpn_outs, targets)

def yolox_loss(outputs, targets):
    if outputs.dtype == torch.float16:
        targets = targets.half()
    loss_cls = F.binary_cross_entropy(outputs[:, :, 5:], targets[:, :, 5:])
    loss_objs = F.binary_cross_entropy(outputs[:, :, 4], targets[:, :, 4])
    return loss_cls.sum() + loss_objs.sum()

def yolox_target_generator(outputs):
    obj_threshold  = 0.5
    cls_threshold  = 0.5
    with torch.no_grad():
        
        objs_targets = (outputs[:, :, 4] > obj_threshold).float().unsqueeze(dim=2)
        cls_targets = (outputs[:, :, 5:] > cls_threshold).float()
        return torch.cat((outputs[:, :, :4], objs_targets, cls_targets), dim=2)
   
