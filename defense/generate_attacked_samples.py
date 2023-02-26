from attack.fgsm import FGSM
from model.custom_yolo import yolox_loss, yolox_target_generator
import cv2
import os
from model.speed_limit_detector import get_model
import torch
from high_level_guided_denoiser import COCODataset
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_attacked_samples(dataloader):
    attack = FGSM()

    attack.model = get_model(device) 
    attack.loss = yolox_loss
    attack.target_generator = yolox_target_generator
    for idx, (input,targets) in enumerate(dataloader):

        input = input.to(device)
        outputs = attack.generate_attack(input)

        for pic_idx, target in enumerate(targets):
            # print(target)
            
            cv2.imwrite(os.path.join(os.path.dirname(os.getcwd()),'model','datasets','attacked_images','val',str(target)),outputs[pic_idx].cpu().numpy().transpose((1,2,0)))

dataset_path = os.path.join(os.path.dirname(os.getcwd()),'model','datasets','tsinghua_gtsdb_speedlimit')
annotations_path = os.path.join(dataset_path,'annotations')
train_dataset = COCODataset(os.path.join(dataset_path,'val2017'),os.path.join(annotations_path,'val2017.json'))
train_dataloader = DataLoader(train_dataset,batch_size=16,pin_memory=True)
generate_attacked_samples(train_dataloader)
