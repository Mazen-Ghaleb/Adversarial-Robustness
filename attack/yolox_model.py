import torch
from onnx2torch import convert
from pathlib import Path
from typing import Union
from torch import Tensor
from torch.nn import  BCELoss
import torch.nn.functional as Func
import onnx

class YoloxModel(torch.nn.Module):
    def __init__(
        self,
        onnx_path: Union[Path, str],
        input_size=[640, 640],
        num_classes:int=10,
        obj_threshold:float = 0.3,
        cls_threshold:float = 0.8
    ) -> None:

        super().__init__()
        temp = onnx.load_model(onnx_path)
        self.model = convert(temp)
        self.bce_loss = BCELoss(reduction="none")
        self.input_size = input_size
        self.num_classes = num_classes
        self.obj_threshold = obj_threshold
        self.cls_threshold = cls_threshold

    def forward(self, x: Tensor, targets=None):
        outputs = self.model(x)
        if self.training:
            cls_pred = outputs[:, :, 5:]
            objs_pred = outputs[:, :, 4]
            if targets is None:
                with torch.no_grad():
                    objs_targets = (objs_pred > self.obj_threshold).float()
                    cls_targets = (cls_pred > self.cls_threshold).float()
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
        cls_loss = self.bce_loss(cls_preds, cls_targets)
        objs_loss = self.bce_loss(objs_preds, objs_targets)

        return cls_loss.sum() + objs_loss.sum()

    def set_thresholds(self, obj_thresholds, cls_thersholds):
        self.obj_threshold = obj_thresholds
        self.cls_threshold = cls_thersholds