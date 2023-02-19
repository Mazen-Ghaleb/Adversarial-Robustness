import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
from pycocotools.coco import COCO
import torch.utils.data as data
import os
import cv2
import torch.optim as optim
import numpy as np

"""
    implementation for high-level representation guided denoiser from
    https://openaccess.thecvf.com/content_cvpr_2018/html/Liao_Defense_Against_Adversarial_CVPR_2018_paper.html
"""
class HGD(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super(HGD, self).__init__()
        ## forward_path
        self.forward_c2 = C2(2, in_channels = in_channels, out_channels=64, kernel_size=3)
        self.forward_c3_1 = C3(3, in_channels=64, out_channels=128, kernel_size=3)
        self.forward_c3_2 = C3(3, in_channels=128, out_channels=256, kernel_size=3)
        self.forward_c3_3 = C3(3, in_channels=256, out_channels=256, kernel_size=3)
        self.forward_c3_4 = C3(3, in_channels=256, out_channels=256, kernel_size=3)

        #backward path
        self.backward_fuse = Fuse()
        self.backward_c3_1 = C3(3, in_channels=512, out_channels=256, kernel_size=3)

        # self.backward_fuse_2 = Fuse()
        self.backward_c3_2 = C3(3, in_channels=512, out_channels=256, kernel_size=3)

        # self.backward_fuse_3 = Fuse()
        self.backward_c3_3 = C3(3, in_channels=384, out_channels=128, kernel_size=3)

        # self.backward_fuse_4 = Fuse()
        self.backward_c2 = C2(2, in_channels=192, out_channels=64, kernel_size=3)
        
        self.conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1)

    def forward(self, x) -> None:
        #forward path
        out_forward_c2 = self.forward_c2(x)
        out_forward_c3_1 = self.forward_c3_1(out_forward_c2)
        out_forward_c3_2 = self.forward_c3_2(out_forward_c3_1)
        out_forward_c3_3 = self.forward_c3_3(out_forward_c3_2)
        out_forward_c3_4 = self.forward_c3_4(out_forward_c3_3)

        #backward path

        out_backward = self.backward_fuse(out_forward_c3_4, out_forward_c3_3)
        out_backward = self.backward_c3_1(out_backward)
        out_backward = self.backward_fuse(out_backward, out_forward_c3_2)
        out_backward = self.backward_c3_2(out_backward)
        out_backward = self.backward_fuse(out_backward, out_forward_c3_1)
        out_backward = self.backward_c3_3(out_backward)
        out_backward = self.backward_fuse(out_backward, out_forward_c2)
        out_backward = self.backward_c2(out_backward)
        out_backward = self.conv(out_backward)

        return out_backward





class C(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding=1):
        super(C, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,  x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class C2(nn.Module):
    def __init__(self, k, in_channels, out_channels, kernel_size) -> None:
        super(C2, self).__init__()
        self.c_blocks = [
            C(in_channels, out_channels, kernel_size),
            C(out_channels, out_channels, kernel_size),
        ]

    def forward(self, x):
        # print("start", x.shape)
        for c_block in self.c_blocks:
            x = c_block(x)
            # print(f"c2: {x.shape}")
        # print("end", x.shape)
        return x

class C3(nn.Module):
    def __init__(self, k, in_channels, out_channels, kernel_size) -> None:
        super(C3, self).__init__()
        self.c_blocks = [
            C(in_channels, out_channels, kernel_size, stride=2),
            C(out_channels, out_channels, kernel_size, stride=1),
            C(out_channels, out_channels, kernel_size, stride=1),
        ]

    def forward(self, x):
        # print("start", x.shape)
        for c_block in self.c_blocks:
            x = c_block(x)
        #     print(f"c3: {x.shape}")
        # print("end", x.shape)
        return x
        


class Fuse(nn.Module):
    def __init__(self) -> None:
        super(Fuse, self).__init__()

    def forward(self, small_image, large_image):
        upscaled_image = F.interpolate(small_image, size=large_image.shape[2:], mode="bilinear")
        result_image = torch.cat((upscaled_image, large_image), dim=1)
        # print(f"fuse: {result_image.shape}")
        return result_image 

class COCODataset(data.Dataset):
    def __init__(self, root_dir,annotaiton_file, transforms=None):
        self.coco = COCO(annotaiton_file)
        self.root_dir = root_dir
        self.transofms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]

        # Load the image 
        img_info  = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if self.transofms is not None:
            img, ann = self.transofms(img, anns)
        return img, anns

class ExperimentalLoss(nn.Module):
    def __init__(self):
        super(ExperimentalLoss,self).__init__()

    def forward(self,denoised_output,bengin_output):
        loss = torch.norm(torch.abs(denoised_output-bengin_output),p=1)
        return loss

class Preprocessor:    
    def preprocess_model_input(self, img, input_size=[640, 640], swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones(
                (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        """TODO::
        this gives correct bounding box  only for images with the same size 
        if you want to use images with different size you must return the ratio for each image 
        and pass it to the output decoder to get the correct box
        """
        self.ratio = min(input_size[0] / img.shape[0],
                            input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * self.ratio), int(img.shape[0] * self.ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * self.ratio),
                    : int(img.shape[1] * self.ratio)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

        return padded_img


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimzer, criterion = ExperimentalLoss()) -> None:
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimzer = optimzer
        self.criterion = criterion

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        for i, (inputs, targets) in enumerate(self.train_loader):
            self.optimzer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimzer.step()
            train_loss += loss.item() * inputs.size(0)
        epoch_train_loss = train_loss / len(self.train_loader.dataset)
        return epoch_train_loss

    def val_epoch(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_loader):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
            epoch_val_loss = val_loss / len(self.val_loader.dataset)
        return epoch_val_loss

    def train(self, n_epochs):
        train_losses = []
        val_losses = []
        for epoch in range(n_epochs):
            epoch_train_loss = self.train_epoch()
            epoch_val_loss = self.val_epoch()
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            print(f"Epoch {epoch + 1}/{n_epochs}, "
                  f"Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, ")
        return train_losses, val_losses 




    
if __name__ == "__main__":
    from torchsummary import summary
    # hgd = HGD()
    # print(sum(p.numel() for p in hgd.parameters() if p.requires_grad))
    
    model = HGD()
    model.eval()
    #model()
    
    from prettytable import PrettyTable

    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
    count_parameters(model)
    
    # TODO: Do the loaders stuff
    #trainer = Trainer(model,TEMP_TRAIN_LOADER,TEMP_VAL_LOADER,optim.Adam(model.parameters(),lr= 0.001))
    

