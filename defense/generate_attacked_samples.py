from attack.fgsm import FGSM
from model.custom_yolo import yolox_loss, yolox_target_generator
import cv2
import os
from model.speed_limit_detector import get_model
import torch
from high_level_guided_denoiser import COCODataset
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from high_level_guided_denoiser import Preprocessor

from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_attacked_samples(dataloader, split_name):
    attack = FGSM()

    attack.model = get_model(device)
    attack.loss = yolox_loss
    attack.target_generator = yolox_target_generator
    for idx, (input, targets) in enumerate(tqdm(dataloader)):

        input = input.to(device)
        outputs = attack.generate_attack(input)

        for pic_idx, target in enumerate(targets):
            # print(target)
            cv2.imwrite(os.path.join(os.path.dirname(os.getcwd()), 'model', 'datasets', 'attacked_images',
                        split_name, str(target)), outputs[pic_idx].cpu().numpy().transpose((1, 2, 0)))


class COCODataset(Dataset):
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
        img = Preprocessor().preprocess_model_input(img)
        
        if self.transofms is not None:
            img, annotation = self.transofms(img, annotation)
        return img, img_info['file_name']

if __name__ == "__main__":
    datasets_path = os.path.join(os.path.dirname(os.getcwd()), 'model', 'datasets')
    os.makedirs(os.path.join(datasets_path, 'attacked_images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(datasets_path, 'attacked_images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(datasets_path, 'attacked_images', 'test'), exist_ok=True)

    dataset_path = os.path.join(os.path.dirname(
        os.getcwd()), 'model', 'datasets', 'tsinghua_gtsdb_speedlimit')
    annotations_path = os.path.join(dataset_path, 'annotations')
    train_dataset = COCODataset(os.path.join(
        dataset_path, 'train2017'), os.path.join(annotations_path, 'train2017.json'))
    train_dataloader = DataLoader(train_dataset, batch_size=1, pin_memory=True)

    test_dataset = COCODataset(os.path.join(
        dataset_path, 'test2017'), os.path.join(annotations_path, 'test2017.json'))
    test_dataloader = DataLoader(train_dataset, batch_size=1, pin_memory=True)

    val_dataset = COCODataset(os.path.join(
        dataset_path, 'val2017'), os.path.join(annotations_path, 'val2017.json'))
    val_dataloader = DataLoader(train_dataset, batch_size=1, pin_memory=True)

    generate_attacked_samples(train_dataloader, 'train')
    generate_attacked_samples(val_dataloader, 'val')
    generate_attacked_samples(test_dataloader, 'test')