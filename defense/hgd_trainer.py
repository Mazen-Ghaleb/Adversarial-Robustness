import torch
import torch.nn as nn 
import torch.nn.functional as F
from pycocotools.coco import COCO
import torch.utils.data as data
import os
import cv2
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from model.speed_limit_detector import get_model
from attack.fgsm import FGSM
from torch.utils.checkpoint import checkpoint
import math
from tqdm import tqdm
from torchviz import make_dot
from yolox.models import IOUloss
from defense.high_level_guided_denoiser import HGD as HGD2
import pandas as pd
"""
    implementation for high-level representation guided denoiser from
    https://openaccess.thecvf.com/content_cvpr_2018/html/Liao_Defense_Against_Adversarial_CVPR_2018_paper.html
"""
class HGD(nn.Module):
    def __init__(self, in_channels=3, width=1) -> None:
        super(HGD, self).__init__()
        ## forward_path
        
        self.forward_c2 = C2(in_channels = in_channels, out_channels=int(64 * width), kernel_size=3)
        self.forward_c3_1 = C3(in_channels=int(64 * width), out_channels=int(128 * width), kernel_size=3)
        self.forward_c3_2 = C3(in_channels=int(128 * width), out_channels=int(256 * width), kernel_size=3)
        self.forward_c3_3 = C3(in_channels=int(256 * width), out_channels=int(256 * width), kernel_size=3)
        self.forward_c3_4 = C3(in_channels=int(256 * width), out_channels=int(256 * width), kernel_size=3)

        #backward path
        self.backward_fuse = Fuse()
        self.backward_c3_1 = C3(in_channels=int(512* width), out_channels=int(256 * width), kernel_size=3)

        # self.backward_fuse_2 = Fuse()
        self.backward_c3_2 = C3(in_channels=int(512 * width), out_channels=int(256 * width), kernel_size=3)

        # self.backward_fuse_3 = Fuse()
        self.backward_c3_3 = C3(in_channels=int(384 * width), out_channels=int(128 * width), kernel_size=3)

        # self.backward_fuse_4 = Fuse()
        self.backward_c2 = C2(in_channels=int(192 * width), out_channels=int(64 * width), kernel_size=3)
        
        self.conv = nn.Conv2d(in_channels=int(64 * width), out_channels=3, kernel_size=1, stride=1)

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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,  x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        #x = checkpoint(self.relu,x)
        return x

class C2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super(C2, self).__init__()
        self.c_blocks = nn.Sequential(
            C(in_channels, out_channels, kernel_size),
            C(out_channels, out_channels, kernel_size),
        )

    def forward(self, x):
        return self.c_blocks(x)

class C3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super(C3, self).__init__()
        self.c_blocks = nn.Sequential(
            C(in_channels, out_channels, kernel_size, stride=2),
            C(out_channels, out_channels, kernel_size, stride=1),
            C(out_channels, out_channels, kernel_size, stride=1),
        )

    def forward(self, x):
        return self.c_blocks(x)
        


class Fuse(nn.Module):
    def __init__(self) -> None:
        super(Fuse, self).__init__()

    def forward(self, small_image, large_image):
        upscaled_image = F.interpolate(small_image, size=large_image.shape[2:], mode="bilinear")
        result_image = torch.cat((upscaled_image, large_image), dim=1)
        return result_image 

class COCODataset(data.Dataset):
    def __init__(
            self,
            model_outputs_path,
            csv_file_path,
            attacked_images_path,
            transforms=None
            ):
        #self.coco = COCO(annotaiton_file)
        self.model_outputs_path = model_outputs_path
        self.transofms = transforms
        self.attacked_images_path = attacked_images_path
        df = pd.read_csv(csv_file_path)
        self.attacked_images = df['attacked_images']
        self.benign_model_outputs = df['benign_model_outputs']
        #self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.attacked_images)

    def __getitem__(self, index):

        attacked_image_name = self.attacked_images[index]
        attacked_image_path = os.path.join(self.attacked_images_path,attacked_image_name)
        attacked_image = cv2.imread(attacked_image_path).transpose((2,0,1))
        attacked_image = np.asarray(attacked_image,dtype=np.float32)

        model_output_path = os.path.join(self.model_outputs_path,
                                          self.benign_model_outputs[index])

        model_output = np.load(model_output_path, mmap_mode='r+')


          
        if self.transofms is not None:
            img, attacked_image = self.transofms(img, attacked_image)
        return attacked_image, model_output

class ExperimentalLoss(nn.Module):
    def __init__(self):
        super(ExperimentalLoss,self).__init__()
        self.iou = IOUloss()

    def forward(self,denoised_output,benign_output):
        
        denoised_output = denoised_output.type(torch.float32)
        benign_output = benign_output.type(torch.float32)
        #cls_loss = F.binary_cross_entropy(denoised_output[:,:,5:],benign_output[:,:,5:]).sum()
        #obj_loss = F.binary_cross_entropy(denoised_output[:,:,4:5],benign_output[:,:,4:5]).sum()
        #combined_loss = torch.linalg.norm(denoised_output[:,:,4:] - benign_output[:,:,4:],dim=(1,2),ord=1).mean()
        # iou_loss = self.iou(denoised_output[:,:,:4],benign_output[:,:,:4]).mean()
        #loss = combined_loss
        # mse_loss = F.mse_loss(benign_output[:, : , :4], denoised_output[:, :, :4])
        denoised_output[:,:,:4] = denoised_output[:,:,:4]/640
        benign_output[:,:,:4] = benign_output[:,:,:4]/640
        
        loss = torch.linalg.norm((denoised_output-benign_output),
                                 dim=(1,2),ord=2).mean()
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
    def __init__(
            self,
            model,
            target_model,
            train_loader,
            val_loader,
            device,
            optimizer,
            criterion,
            scheduler,
            fp16=True
            ) -> None:
        self.model = model
        self.target_model = target_model
        self.device = device
        self.model.to(self.device)
        self.scheduler = scheduler
        self.target_model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        #self.attack = FGSM()
        # self.attack.model = self.target_model
        self.best_val_loss = math.inf
        self.fp16 = fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.data_type = torch.float16 if self.fp16 else torch.float32
        self.__disable_target_model_wieghts_grad()

    def __disable_target_model_wieghts_grad(self):
        for parameter in self.target_model.parameters():
            parameter.requires_grad = False

    def __print_params_grad(self):
        for name, parameter in self.target_model.named_parameters():
            print(name, parameter.requires_grad)

        for name, parameter in self.model.named_parameters():
            print(name, parameter.requires_grad)

    def __visualize(self, value, parameters, name, format="pdf"):
        dot = make_dot(value, parameters)
        dot.render(name, format=format)

    def train_epoch(self,epoch):
        print(f"Now training epoch {epoch}")
        self.model.train()
        train_loss = 0.0
        pbar = tqdm(self.train_loader, desc='')
        for i, (perturbed_images, target_model_targets) in enumerate(pbar):

            with torch.cuda.amp.autocast(enabled=self.fp16):
                target_model_targets = target_model_targets.to(self.data_type).to(self.device)

                perturbed_images = perturbed_images.to(self.data_type).to(self.device)
                hgd_outputs = self.model(perturbed_images)
                self.target_model.eval()
                denoised_images = perturbed_images - hgd_outputs
                target_model_outputs = self.target_model(denoised_images)
                loss = self.criterion(target_model_outputs, target_model_targets)

            # with torch.no_grad():
            #     target_model_targets = self.target_model(images)
                    #target_model_targets = torch.randn(target_model_outputs.shape).half().to(self.device)
            #target_model_outputs = torch.randn(target_model_targets.shape).to(self.device)
            
            
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            pbar.set_description(f"Training Loss: {loss.item():.4f}")
            # self.optimzer.step()
            train_loss += loss.item() * target_model_targets.size(0)
        
        epoch_train_loss = train_loss / len(self.train_loader.dataset)
        print(f"Training for epoch number {epoch} resulted in epoch_train_loss = {epoch_train_loss}")
        return epoch_train_loss

    def val_epoch(self,epoch):
        print(f'Running Validation for epoch {epoch}')
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(self.val_loader)
            for i, (perturbed_images, target_model_targets) in enumerate(pbar):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    target_model_targets = target_model_targets.to(self.data_type).to(self.device)
                    perturbed_images = perturbed_images.to(self.data_type).to(self.device)
                    hgd_outputs = self.model(perturbed_images)
                    
                    #hgd_outputs = hgd_outputs.type(torch.float32)
                    denoised_images = perturbed_images - hgd_outputs
                    target_model_outputs = self.target_model(denoised_images)
                    loss = self.criterion(target_model_outputs, target_model_targets)
                    pbar.set_description(f"Validation Loss: {loss.item():.4f}")
                    val_loss += loss.item() * target_model_targets.size(0)
            epoch_val_loss = val_loss / len(self.val_loader.dataset)
        print(f'Validation for epoch number {epoch} resulted in epoch_val_loss = {epoch_val_loss}')
        if (epoch_val_loss < self.best_val_loss):
            torch.save({'model_dict':self.model.state_dict(),
                        'epoch': epoch},"best_ckpt.pt")
            self.best_val_loss = epoch_val_loss
        return epoch_val_loss

    def train(self, n_epochs):
        train_losses = []
        val_losses = []
        for epoch in range(n_epochs):
            epoch_train_loss = self.train_epoch(epoch)
            epoch_val_loss = self.val_epoch(epoch)
            self.scheduler.step()
            #print(f"Epoch number {epoch}/{n_epochs}, train_loss: {epoch_train_loss}, val_loss: {epoch_val_loss}")
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            print(f"Epoch {epoch + 1}/{n_epochs},"
                  f"Train Loss: {epoch_train_loss:.4f},"
                  f"Val Loss: {epoch_val_loss:.4f},")
        return train_losses, val_losses 


def get_HGD_model(device):
    dir_relative_path = os.path.relpath(os.path.dirname(__file__), os.getcwd())
    # get the path of the model and the expirement script
    model_path = os.path.join(dir_relative_path, "best_ckpt.pt")
    model = HGD(width=0.5)
    # model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.load_state_dict(torch.load(model_path))
    return model.to(device)


    
if __name__ == "__main__":
    from torchsummary import summary
    np.random.seed(42)
    # print(sum(p.numel() for p in hgd.parameters() if p.requires_grad))
    
    # model = HGD(width=0.5)
    model = HGD2(width=1, growth_rate=32, bn_size=4)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from prettytable import PrettyTable

    def count_parameters(model, input_size):
        from torchsummary import summary
        summary(model, input_size)
    # count_parameters(model, (3, 640, 640))
    # count_parameters(model, (3, 224, 224))
    
    
    

    dataset_path = os.path.join(
        os.path.dirname(os.getcwd()),'model','datasets','tsinghua_gtsdb_speedlimit')
    model_outputs_path= os.path.join(
        os.path.dirname(os.getcwd()),'model','datasets','model_outputs')
    attacked_images_path = os.path.join(
        os.path.dirname(os.getcwd()),'model','datasets','attacked_images')

    
        
    annotations_path = os.getcwd() #os.path.join(dataset_path,'annotations')
    train_dataset = COCODataset(
        os.path.join(model_outputs_path,'train'),
        os.path.join(attacked_images_path,'train.csv'),
        os.path.join(attacked_images_path,'train'))
    # test_dataset = COCODataset(os.path.join(dataset_path,'test2017'),os.path.join(annotations_path,'test2017.json'))
    val_dataset = COCODataset(
        os.path.join(model_outputs_path,'val'),
        os.path.join(attacked_images_path,'val.csv'),
        os.path.join(attacked_images_path,'val'))
    

    batch_size_train = 8
    batch_size_val = 32
    train_dataloader = DataLoader(train_dataset,batch_size= batch_size_train,
                                   shuffle=True,pin_memory=True,num_workers=2,prefetch_factor=5)
    val_dataloader = DataLoader(val_dataset,batch_size= batch_size_val,
                                 shuffle=True,pin_memory=True,num_workers=2,prefetch_factor=5)

    
    
    # print(len(train_dataset))

    
    #train_dataset.__getitem__(0)

    
    target_model = get_model(device)

    # optimizer = optim.SGD(model.parameters(),lr= 1e-5,momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model,
        target_model,
        train_dataloader,
        val_dataloader,
        device,optimizer,
        criterion= ExperimentalLoss(),
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.97),
        fp16=True)

    trainer.train(300)
    # trainer.val_epoch(1)
    