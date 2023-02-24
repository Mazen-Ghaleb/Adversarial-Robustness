from YOLOX import YOLOXHead
import torch
from torch import Tensor
import torch.nn.functional as F


class CustomYOLOHead(YOLOXHead):
    def __init__(
        self,
        num_classes,
        width=1,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        obj_threshold=0.8,
        cls_threshold=0.8):
        super().__init__(num_classes, width, strides, in_channels, act, depthwise)

        self.obj_threshold = obj_threshold
        self.cls_threshold=cls_threshold

    def forward(self, xin, labels=None, imgs=None):
        outputs = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )
            outputs.append(output)

        # [batch, n_anchors_all, 85]
        outputs = torch.cat(
            [x.flatten(start_dim=2) for x in outputs], dim=2
        ).permute(0, 2, 1)
        return outputs