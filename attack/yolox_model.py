import torch
from onnx2torch import convert
from pathlib import Path
from typing import Union
from torch import Tensor
from torch.nn import  BCEWithLogitsLoss
import torch.nn.functional as Func


class YoloxModel(torch.nn.Module):
    def __init__(
        self,
        onnx_path: Union[Path, str],
        input_size=[640, 640],
        num_classes:int=10
    ) -> None:

        super().__init__()
        self.model = convert(onnx_path)
        self.cls_loss = BCEWithLogitsLoss(reduction="none")
        self.input_size = input_size
        self.num_classes = num_classes

    def forward(self, x: Tensor, targets=None):
        outputs = self.model(x)
        if self.training:
            cls_pred = outputs[:, :, 5:]
            objs_pred = outputs[:, :, 4].view(1, -1, 1)
            if targets is None:
                max_cls = torch.argmax(cls_pred, 2)
                max_objs = torch.argmax(objs_pred, 2)
                cls_targets = Func.one_hot(
                    max_cls, num_classes=self.num_classes).to(torch.float32)
                objs_targets = Func.one_hot(
                    max_objs, num_classes=1
                ).to(torch.float32)
            loss = self.get_cls_loss(cls_pred,objs_pred,cls_targets, objs_targets)
            return loss
        else:
            return outputs

    def get_cls_loss(
            self,
            cls_preds: Tensor,
            objs_preds: Tensor,
            cls_targets: Tensor,
            objs_targets: Tensor):
        cls_loss = self.cls_loss(cls_preds, cls_targets)
        objs_loss = self.cls_loss(objs_preds, objs_targets)

        return cls_loss.sum() + objs_loss.sum()

