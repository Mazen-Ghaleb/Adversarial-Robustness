from YOLOX import YOLOX as BaseYOLOX

class CustomYOLOX(BaseYOLOX):
    def __init__(self, backbone=None, head=None):
        super().__init__(backbone, head)

    def forward(self, x, targets=None):
        fpn_outs = self.backbone(x)
        return self.head(fpn_outs, targets)