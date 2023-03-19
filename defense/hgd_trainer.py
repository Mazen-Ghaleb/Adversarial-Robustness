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
from tqdm.auto import tqdm
from torchviz import make_dot
from yolox.models import IOUloss
from defense.high_level_guided_denoiser import HGD as HGD2
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
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
            orig_images_path,
            csv_file_path,
            attacked_images_path,
            transforms=None
            ):
        #self.coco = COCO(annotaiton_file)
        self.orig_images_path = orig_images_path
        self.transofms = transforms
        self.attacked_images_path = attacked_images_path
        df = pd.read_csv(csv_file_path)
        self.attacked_images = df['attacked_images']
        self.orig_images = df['orig_images']
        self.preprocessor = Preprocessor()
        #self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.attacked_images)

    def __getitem__(self, index):

        attacked_image_name = self.attacked_images[index]
        orig_image_name = self.orig_images[index]

        attacked_image_path = os.path.join(self.attacked_images_path,attacked_image_name)
        orig_image_path = os.path.join(self.orig_images_path,orig_image_name)

        attacked_image = cv2.imread(attacked_image_path).transpose((2,0,1))
        # attacked_image = np.asarray(attacked_image,dtype=np.float32)

        orig_image = cv2.imread(orig_image_path)
        orig_image = self.preprocessor.preprocess_model_input(orig_image)

        # model_output_path = os.path.join(self.model_outputs_path,
        #                                   self.benign_model_outputs[index])

        # model_output = np.load(model_output_path, mmap_mode='r+')
        # features = (model_output['p3'], model_output['p4'], model_output['p5'])

          
        if self.transofms is not None:
            img, attacked_image = self.transofms(img, attacked_image)
        return attacked_image, orig_image

class ExperimentalLoss(nn.Module):
    def __init__(self, regularization_factor=1e-4):
        super(ExperimentalLoss,self).__init__()
        self.regularization_factor = regularization_factor
        self.iou = IOUloss()

    def forward(self,
                denoised_output: torch.Tensor,
                benign_output,
                noise: torch.Tensor,
                denoised_images: torch.Tensor):
        
        # denoised_output = denoised_output.type(torch.float32)
        # benign_output = benign_output.type(torch.float32)

        p3_loss = torch.abs(benign_output[0] - denoised_output[0]).mean()
        p4_loss = torch.abs(benign_output[1] - denoised_output[1]).mean()
        p5_loss = torch.abs(benign_output[2] - denoised_output[2]).mean()
        
        denoised_image_loss = torch.pow(torch.where(torch.logical_or(denoised_images > 255,
                                                                      denoised_images < 0),
                                          denoised_images,
                                          torch.zeros_like(denoised_images)), 2).mean()
        denoised_image_loss *= self.regularization_factor

        noise_loss = torch.pow(torch.where(torch.logical_or(noise > 255, noise < -255),
                                          noise,
                                          torch.zeros_like(noise))
                                          , 2).mean()
        noise_loss *= self.regularization_factor

        total_loss = p3_loss + p4_loss + p5_loss +  denoised_image_loss + noise_loss

        losses = {
            "noise_loss": noise_loss,
            "denoised_images_loss": denoised_image_loss,
            "p3_loss": p3_loss,
            "p4_loss": p4_loss,
            "p5_loss": p5_loss,
            "total_loss": total_loss,
            }
        
        return losses 

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
            fp16=True,
            accumlation_steps = 4,
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
        self.best_val_loss = math.inf
        self.fp16 = fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.data_type = torch.float16 if self.fp16 else torch.float32
        self.accumlation_steps = accumlation_steps
        self.writer = SummaryWriter()

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
    
    def write_to_tensorboard(self, losses, epoch, split):
        for key in losses:
            self.writer.add_scalar(f'{split}_{key}', losses[key].item(), epoch)

    def increment_loss(self, losses, losses_to_add, size):
        for key in losses:
            losses[key] += losses_to_add[key] * size
    
    def normalize_loss(self, losses, dividor):
        for key in losses:
            losses[key] /= dividor
    
    

    def train_epoch(self,epoch):
        train_bpar = tqdm(enumerate(self.train_loader),
                          initial=1, total = len(self.train_loader),leave=None)
        train_bpar.set_description(f'train_loss: ')
        self.model.train()

        total_norm_params = 0.0
        total_norm_grads = 0.0

        epoch_losses = {
            "noise_loss": 0.0,
            "denoised_images_loss": 0.0,
            "p3_loss": 0.0,
            "p4_loss": 0.0,
            "p5_loss": 0.0,
            "total_loss": 0.0,
            }
        num_params = sum(p.numel() for p in self.model.parameters())

        for i, (perturbed_images, orig_images) in train_bpar:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                orig_images = orig_images.to(self.data_type).to(self.device)

                perturbed_images = perturbed_images.to(self.data_type).to(self.device)
                noise = self.model(perturbed_images)
                self.target_model.eval()
                denoised_images = perturbed_images - noise
                target_model_outputs = self.target_model(denoised_images)

                with torch.no_grad():
                    target_model_targets = self.target_model(orig_images)

                losses = self.criterion(target_model_outputs,
                                        target_model_targets,
                                        noise,
                                        denoised_images) 

                loss = losses["total_loss"]
                # self.__visualize(loss, dict(self.model.named_parameters()) |
                #                   dict(self.target_model.named_parameters()), "Train Loop6")
                train_bpar.set_description(f'train_loss: {loss.item():.4f}')
            
            # self.__visualize(loss, dict(self.target_model.named_parameters()) |
            #                  dict(self.model.named_parameters()), "Features Loop")


            self.scaler.scale(loss/self.accumlation_steps).backward()
        
            if ((i + 1) % self.accumlation_steps) == 0 or i == len(self.train_loader) - 1:
                with torch.no_grad():
                    total_norm_params += torch.norm(
                        torch.cat([p.grad.flatten() for p in self.model.parameters()]), p=2).item()
                    total_norm_grads += torch.norm(
                        torch.cat([p.flatten() for p in self.model.parameters()]), p=2).item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            self.increment_loss(epoch_losses,losses,size= perturbed_images.shape[0])

        self.normalize_loss(epoch_losses, len(self.train_loader.dataset))

        epoch_avg_norm_params = total_norm_params / num_params
        epoch_avg_norm_grads = total_norm_grads / num_params

        self.writer.add_scalar('avg_norm_params', epoch_avg_norm_params, global_step=epoch + 1)
        self.writer.add_scalar('avg_norm_grads', epoch_avg_norm_grads, global_step=epoch + 1)

        return epoch_losses
    
    def save_checkpoint(self, epoch_losses, epoch):
        if (epoch_losses['total_loss'].item() < self.best_val_loss):
            torch.save({'model_dict':self.model.state_dict(),
                        'epoch': epoch},"best_ckpt.pt")
            self.best_val_loss = epoch_losses['total_loss'].item()

    @torch.no_grad()
    def val_epoch(self,epoch):
        epoch_losses = {
            "noise_loss": 0.0,
            "denoised_images_loss": 0.0,
            "p3_loss": 0.0,
            "p4_loss": 0.0,
            "p5_loss": 0.0,
            "total_loss": 0.0,
        }

        self.model.eval()
        val_bpar = tqdm(self.val_loader, initial=1, total=len(self.val_loader),leave=None)
        val_bpar.set_description('val_loss:')
        for perturbed_images, orig_images in val_bpar:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                orig_images = orig_images.to(self.data_type).to(self.device)
                perturbed_images = perturbed_images.to(self.data_type).to(self.device)
                noise = self.model(perturbed_images)
                
                #hgd_outputs = hgd_outputs.type(torch.float32)
                denoised_images = perturbed_images - noise
                target_model_outputs = self.target_model(denoised_images)
                target_model_targets = self.target_model(orig_images)
                losses = self.criterion(target_model_outputs, target_model_targets, noise, denoised_images)
                loss =  losses["total_loss"]
                val_bpar.set_description(f"val_loss: {loss:.4f}")
                self.increment_loss(epoch_losses, losses, perturbed_images.shape[0])
        self.normalize_loss(epoch_losses, len(self.val_loader.dataset))
        

        return epoch_losses
    
    def print_losses(self, losses, split):
        for key in losses:
           print(f"{split}_{key}:{losses[key].item():.4f}", end=', ')

        print("")



    def train(self, n_epochs):
        self.no_epochs = n_epochs
        bpar = tqdm(range(n_epochs), initial=1)
        for epoch in bpar:
            bpar.set_description(f"Epoch {epoch + 1}, train_loss: {math.nan}" \
                                 f",val_loss: {math.nan}")

            train_losses = self.train_epoch(epoch)
            bpar.set_description(f"Epoch {epoch + 1}, train_loss: "\
                                 f"{train_losses['total_loss']}" \
                                 f",val_loss: {math.nan}")
            self.print_losses(train_losses, "train")
            val_losses = self.val_epoch(epoch)

            bpar.set_description(f"Epoch {epoch + 1}, train_loss: "\
                                 f"{train_losses['total_loss']}" \
                                 f",val_loss: {val_losses['total_loss']}")
            self.print_losses(val_losses, "val")

            print(f"Epoch number {epoch + 1}/{n_epochs},"\
                  f"train_loss: {train_losses['total_loss']},"\
                  f" val_loss: {val_losses['total_loss']}")

            self.scheduler.step(val_losses['total_loss'])

            self.save_checkpoint(val_losses, epoch)

            self.write_to_tensorboard(train_losses, epoch, 'train')
            self.write_to_tensorboard(val_losses, epoch, 'val')
            self.writer.flush()

def get_HGD_model(device):
    dir_relative_path = os.path.relpath(os.path.dirname(__file__), os.getcwd())
    # get the path of the model and the expirement script
    model_path = os.path.join(dir_relative_path, "best_ckpt.pt")
    model = HGD2(width=1.0, growth_rate=32, bn_size=4)
    # model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.load_state_dict(torch.load(model_path, device)['model_dict'])
    return model.to(device)


    
if __name__ == "__main__":
    from torchsummary import summary
    np.random.seed(42)
    model = HGD2(width=1, growth_rate=32, bn_size=4)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = os.path.join(
        os.path.dirname(os.getcwd()),'model','datasets','tsinghua_gtsdb_speedlimit')
    model_outputs_path= os.path.join(
        os.path.dirname(os.getcwd()),'model','datasets','model_features')
    attacked_images_path = os.path.join(
        os.path.dirname(os.getcwd()),'model','datasets','attacked_images')

    
        
    annotations_path = os.getcwd() #os.path.join(dataset_path,'annotations')
    train_dataset = COCODataset(
        os.path.join(dataset_path,'train2017'),
        os.path.join(attacked_images_path,'train.csv'),
        os.path.join(attacked_images_path,'train'))
    # test_dataset = COCODataset(os.path.join(dataset_path,'test2017'),os.path.join(annotations_path,'test2017.json'))
    val_dataset = COCODataset(
        os.path.join(dataset_path,'val2017'),
        os.path.join(attacked_images_path,'val.csv'),
        os.path.join(attacked_images_path,'val'))
    

    batch_size_train = 8
    batch_size_val = 32
    num_workers = 2
    prefetch_factor = 5
    train_dataloader = DataLoader(train_dataset,batch_size= batch_size_train,
                                   shuffle=True,pin_memory=True,num_workers=num_workers,prefetch_factor=prefetch_factor)
    val_dataloader = DataLoader(val_dataset,batch_size= batch_size_val,
                                 shuffle=True,pin_memory=True,num_workers=num_workers,prefetch_factor=prefetch_factor)

    target_model = get_model(device).backbone

    # optimizer = optim.SGD(model.parameters(),lr= 1e-4,momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    trainer = Trainer(
        model,
        target_model,
        train_dataloader,
        val_dataloader,
        device,optimizer,
        criterion= ExperimentalLoss(regularization_factor=1e-4),
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,min_lr=5e-5,factor=0.8,
                                                         patience=5,mode='min'),
        fp16=True,
        accumlation_steps = 16,
        )
    trainer.train(300)
    # trainer.val_epoch(1)
    