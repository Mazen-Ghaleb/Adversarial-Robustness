from glob import glob
import os
from tqdm import tqdm
import pandas as pd

attacked_images_dataset_path = os.path.join(os.path.dirname(os.getcwd()),'model','datasets','attacked_images')

attacked_images_train = glob(os.path.join(attacked_images_dataset_path,'train','*'))
attacked_images_test = glob(os.path.join(attacked_images_dataset_path,'test','*'))
attacked_images_val = glob(os.path.join(attacked_images_dataset_path,'val','*'))

train_dict = {"attacked_images": [],"benign_model_outputs":[]}
test_dict = {"attacked_images": [],"benign_model_outputs":[]}
val_dict = {"attacked_images": [],"benign_model_outputs":[]}

for image_path in tqdm(attacked_images_train):
    image_name = os.path.basename(image_path)
    splits = image_name.split(".")
    image_id = splits[0]
    
    train_dict['attacked_images'].append(image_name)
    train_dict['benign_model_outputs'].append(f"{image_id}.npy")
    
df = pd.DataFrame(train_dict)
df.to_csv(os.path.join(attacked_images_dataset_path,'train.csv'),index=None)

for image_path in tqdm(attacked_images_test):
    image_name = os.path.basename(image_path)
    splits = image_name.split(".")
    image_id = splits[0]
    
    test_dict['attacked_images'].append(image_name)
    test_dict['benign_model_outputs'].append(f"{image_id}.npy")
    
df = pd.DataFrame(test_dict)
df.to_csv(os.path.join(attacked_images_dataset_path,'test.csv'),index=None)

for image_path in tqdm(attacked_images_val):
    image_name = os.path.basename(image_path)
    splits = image_name.split(".")
    image_id = splits[0]
    
    val_dict['attacked_images'].append(image_name)
    val_dict['benign_model_outputs'].append(f"{image_id}.npy")
    
df = pd.DataFrame(val_dict)
df.to_csv(os.path.join(attacked_images_dataset_path,'val.csv'),index=None)