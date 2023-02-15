import os
from yolox.exp import Exp as MyExp

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

        self.num_classes = 10

        self.warmup_epochs = 4
        self.max_epoch = 50
        self.data_num_workers = 8

        self.print_interval = 20
        self.eval_interval = 1

        self.input_size = (640, 640)
        self.multiscale_range = 0