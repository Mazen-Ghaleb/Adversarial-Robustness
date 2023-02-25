import os
from yolox.exp import Exp as MyExp
from custom_yolo_head import CustomYOLOHead
import torch.nn as nn
from custom_yolo import CustomYOLOX
import torch

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33 # values for the yolox_s
        self.width = 0.50 # values for the yolox_s
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/tsinghua_gtsdb_speedlimit"
        self.train_ann = "train2017.json"
        self.val_ann = "val2017.json"
        self.test_ann = "test2017.json"

        self.num_classes = 10

        self.warmup_epochs = 4
        self.max_epoch = 50
        self.data_num_workers = 8

        self.print_interval = 20
        self.eval_interval = 1

        self.input_size = (640, 640)
        self.multiscale_range = 0

    def get_model(self):
        from yolox.models import YOLOPAFPN

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = CustomYOLOHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = CustomYOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
    
    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.Adam(
                pg0, lr=lr,
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer