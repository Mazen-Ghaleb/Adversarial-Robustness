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
        
        """
            loss calculation required for the attack
        """
        if self.training:
            """
            If there's no labels given that means we are doing an untargeted attack and need some labels to obtain the 
            loss function so we transform  the output of the model into a one hot vector encoding for the classificatoins
            and simmilarly for the objectness to obtain fake targets and then calculate the loss on them

            TODO: A random box could be added here to attack the part of the bounding box

            """
            if labels is None:
                with torch.no_grad():
                    objs_targets = (outputs[:, :, 4] > self.obj_threshold).float()
                    cls_targets = (outputs[:, :, 5:] > self.cls_threshold).float()
                return self.get_custom_loss(outputs, cls_targets, objs_targets)
        else:
            return outputs

    """ 
    TODO:
    The loss we obtain can be imporved by letting the model decode its output to obtain the number of objects actually present
    in the images 
    check this
    """

    def get_custom_loss(
            self,
            outputs, 
            cls_targets: Tensor,
            objs_targets: Tensor,
            ):

        loss_cls = F.binary_cross_entropy(outputs[:, :, 5:], cls_targets)
        loss_objs = F.binary_cross_entropy(outputs[:, :, 4], objs_targets)
        return loss_cls.sum() + loss_objs.sum()