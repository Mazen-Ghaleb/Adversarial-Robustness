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
from torch.utils.data import DataLoader
# from model.sign_classifier import classifier_loss, classifier_target_generator
from model.custom_yolo import yolox_loss, yolox_target_generator
from model.speed_limit_detector import get_model
from attack.fgsm import FGSM
import math


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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,bias=False)
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
        self.c_blocks = nn.Sequential(
            C(in_channels, out_channels, kernel_size),
            C(out_channels, out_channels, kernel_size),
        )

    def forward(self, x):
        # print("start", x.shape)
        # for c_block in self.c_blocks:
        # x = c_block(x)
            # print(f"c2: {x.shape}")
        # print("end", x.shape)
        return self.c_blocks(x)

class C3(nn.Module):
    def __init__(self, k, in_channels, out_channels, kernel_size) -> None:
        super(C3, self).__init__()
        self.c_blocks = nn.Sequential(
            C(in_channels, out_channels, kernel_size, stride=2),
            C(out_channels, out_channels, kernel_size, stride=1),
            C(out_channels, out_channels, kernel_size, stride=1),
        )

    def forward(self, x):
        # print("start", x.shape)
        # for c_block in self.c_blocks:
        #     x = c_block(x)
        #     print(f"c3: {x.shape}")
        # print("end", x.shape)
        return self.c_blocks(x)
        


class Fuse(nn.Module):
    def __init__(self) -> None:
        super(Fuse, self).__init__()

    def forward(self, small_image, large_image):
        upscaled_image = F.interpolate(small_image, size=large_image.shape[2:], mode="bilinear")
        result_image = torch.cat((upscaled_image, large_image), dim=1)
        # print(f"fuse: {result_image.shape}")
        return result_image 

class COCODataset(data.Dataset):
    def __init__(self, root_dir,annotaiton_file, attacked_images_path, transforms=None):
        self.coco = COCO(annotaiton_file)
        self.root_dir = root_dir
        self.transofms = transforms
        self.attacked_images_path = attacked_images_path
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
        img = Preprocessor().preprocess_model_input(img)
        
        img = np.asarray(img,dtype=np.float32)
        # if img.shape != (3,640,640):
        #     print("FALSE: " + str(img.shape))
        # else:
        #     print("True")
        # ann_ids = coco.getAnnIds(imgIds=img_id)
        # anns = coco.loadAnns(ann_ids)
        
        attacked_image_path = os.path.join(self.attacked_images_path,img_info['file_name'])
        attacked_image = cv2.imread(attacked_image_path).transpose((2,0,1))
        attacked_image = np.asarray(attacked_image,dtype=np.float32)
          
        if self.transofms is not None:
            img, attacked_image = self.transofms(img, attacked_image)
        return img, attacked_image

class ExperimentalLoss(nn.Module):
    def __init__(self):
        super(ExperimentalLoss,self).__init__()

    def forward(self,denoised_output,benign_output):
        
        denoised_output = denoised_output.type(torch.float32)
        benign_output = benign_output.type(torch.float32)
        # print(f"denoised output {denoised_output.dtype}")
        # print(f"benign output {benign_output.dtype}")
        loss = torch.linalg.norm(denoised_output-benign_output,dim=(1,2),ord=2).mean()

        #print(f"loss {loss}")
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
    def __init__(self, model, target_model, train_loader, val_loader, device, optimzer, criterion) -> None:
        self.model = model
        self.target_model = target_model
        self.device = device
        self.model.to(self.device)
        
        self.target_model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimzer = optimzer
        self.criterion = criterion
        self.attack = FGSM()
        self.attack.model = self.target_model
        self.attack.loss = yolox_loss
        self.attack.target_generator = yolox_target_generator
        self.best_val_loss = math.inf

    def train_epoch(self,epoch):
        print(f"Now training epoch {epoch}")
        self.model.train()
        train_loss = 0.0
        for i, (images,perturbed_images) in enumerate(self.train_loader):
            print(f"Training on batch number {i} of {len(self.train_loader)}")
            #images = images.half()
            #perturbed_images = perturbed_images.half()
            
            # perturbed_images = self.attack.generate_attack(images)
            
            self.optimzer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                images = images.to(self.device)

                perturbed_images = perturbed_images.to(self.device)

                hgd_outputs = self.model(perturbed_images)
            

                self.target_model.eval()
                # with torch.no_grad():
                denoised_images = perturbed_images - hgd_outputs


            target_model_outputs = self.target_model(denoised_images)
            
            with torch.no_grad():
                target_model_targets = self.target_model(images)
                    #target_model_targets = torch.randn(target_model_outputs.shape).half().to(self.device)
            
            with torch.autocast('cuda'):
                loss = self.criterion(target_model_outputs, target_model_targets)
            
            loss.backward()
            
            self.optimzer.step()
            train_loss += loss.item() * images.size(0)
        epoch_train_loss = train_loss / len(self.train_loader.dataset)
        print(f"Training for epoch number {epoch} resulted in epoch_train_loss = {epoch_train_loss}")
        return epoch_train_loss

    def val_epoch(self,epoch):
        print(f'Running Validation for epoch {epoch}')
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (images, perturbed_images) in enumerate(self.val_loader):
                print(f"Validating for batch number {i} of {len(self.val_loader)}")
                with torch.autocast('cuda'):
                    images = images.to(self.device)
                    perturbed_images = perturbed_images.to(self.device)
                    hgd_outputs = self.model(perturbed_images)
                    hgd_outputs = hgd_outputs.type(torch.float32)
                    # print(f"images: {images.dtype}")
                    # print(f"perturbed_images: {perturbed_images.dtype}")
                    # print(f"hgd_outputs: {hgd_outputs.dtype}")
                    denoised_images = perturbed_images - hgd_outputs
                    target_model_outputs = self.target_model(denoised_images)
                    target_model_targets = self.target_model(images)

                    loss = self.criterion(target_model_outputs, target_model_targets)
                    val_loss += loss.item() * images.size(0)
            epoch_val_loss = val_loss / len(self.val_loader.dataset)
        print(f'Validation for epoch number {epoch} resulted in epoch_val_loss = {epoch_val_loss}')
        if (epoch_val_loss < self.best_val_loss):
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimzer.state_dict(),
                'loss': epoch_val_loss,
            },"best_ckpt.pt")
            self.best_val_loss = epoch_val_loss
        return epoch_val_loss

    def train(self, n_epochs):
        train_losses = []
        val_losses = []
        for epoch in range(n_epochs):
            epoch_train_loss = self.train_epoch(epoch)
            epoch_val_loss = self.val_epoch(epoch)
            print(f"Epoch number {epoch}/{n_epochs}, train_loss: {epoch_train_loss}, val_loss: {epoch_val_loss}")
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            print(f"Epoch {epoch + 1}/{n_epochs},"
                  f"Train Loss: {epoch_train_loss:.4f},"
                  f"Val Loss: {epoch_val_loss:.4f},")
        return train_losses, val_losses 




    
if __name__ == "__main__":
    from torchsummary import summary
    
    # hgd = HGD()
    # print(sum(p.numel() for p in hgd.parameters() if p.requires_grad))
    
    model = HGD()
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    #count_parameters(model)
    
    
    

    dataset_path = os.path.join(os.path.dirname(os.getcwd()),'model','datasets','tsinghua_gtsdb_speedlimit')
    attacked_images_path = os.path.join(os.path.dirname(os.getcwd()),'model','datasets','attacked_images')
    annotations_path = os.path.join(dataset_path,'annotations')
    train_dataset = COCODataset(os.path.join(dataset_path,'train2017'),os.path.join(annotations_path,'train2017.json'),os.path.join(attacked_images_path,'train'))
    # test_dataset = COCODataset(os.path.join(dataset_path,'test2017'),os.path.join(annotations_path,'test2017.json'))
    val_dataset = COCODataset(os.path.join(dataset_path,'val2017'),os.path.join(annotations_path,'val2017.json'),os.path.join(attacked_images_path,'val'))
    

    
    train_dataloader = DataLoader(train_dataset,batch_size= 4, shuffle=True,pin_memory=True)
    val_dataloader = DataLoader(val_dataset,batch_size= 4, shuffle=True,pin_memory=True)

    
    
    # print(len(train_dataset))

    # for itemidx in range(len(train_dataset)):
    #     train_dataset.__getitem__(itemidx)

    
    target_model = get_model(device)

    optimizer = optim.SGD(model.parameters(),lr= 1e-3,momentum=0.9)
    trainer = Trainer(model,target_model,train_dataloader, val_dataloader, device,optimizer, criterion= ExperimentalLoss())

    trainer.train(50)
    # trainer.val_epoch(0)
    