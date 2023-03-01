from generate_attacked_samples import COCODataset
from model.speed_limit_detector import get_model
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import json 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device)
model.eval()

if __name__ == "__main__":
    datasets_path = os.path.join(os.path.dirname(os.getcwd()), 'model', 'datasets')
    os.makedirs(os.path.join(datasets_path, 'model_outputs', 'train'), exist_ok=True)
    os.makedirs(os.path.join(datasets_path, 'model_outputs', 'val'), exist_ok=True)
    os.makedirs(os.path.join(datasets_path, 'model_outputs', 'test'), exist_ok=True)

    dataset_path = os.path.join(os.path.dirname(
        os.getcwd()), 'model', 'datasets', 'tsinghua_gtsdb_speedlimit')
    annotations_path = os.path.join(dataset_path, 'annotations')
    train_dataset = COCODataset(os.path.join(
        dataset_path, 'train2017'), os.path.join(annotations_path, 'train2017.json'))
    train_dataloader = DataLoader(train_dataset, batch_size=1, pin_memory=True)

    test_dataset = COCODataset(os.path.join(
        dataset_path, 'test2017'), os.path.join(annotations_path, 'test2017.json'))
    test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

    val_dataset = COCODataset(os.path.join(
        dataset_path, 'val2017'), os.path.join(annotations_path, 'val2017.json'))
    val_dataloader = DataLoader(val_dataset, batch_size=1, pin_memory=True)

    train_outputs = {}
    test_outputs = {}
    val_outputs = {}
    
    for i, (inputs,targets) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            inputs = inputs.to(device)
            test_outputs[targets[0]] = model(inputs)[0].cpu().numpy().tolist()
    
    print("Dumping test to JSON")

    with open(os.path.join(datasets_path, 'model_outputs', 'test','test_outputs.json'),"w") as f:
        json.dump(test_outputs,f)      

    for i, (inputs,targets) in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            inputs = inputs.to(device)
            val_outputs[targets[0]] = model(inputs)[0].cpu().numpy().tolist()


    print("Dumping test to JSON")
    
    with open(os.path.join(datasets_path, 'model_outputs', 'val','val_outputs.json'),"w") as f:
        json.dump(val_outputs,f)
    
    for i, (inputs,targets) in enumerate(tqdm(train_dataloader)):
        with torch.no_grad():
            inputs = inputs.to(device)
            train_outputs[targets[0]] = model(inputs)[0].cpu().numpy().tolist()
        
    print("Dumping train to JSON")
    
    with open(os.path.join(datasets_path, 'model_outputs', 'train','train_outputs.json'),"w") as f:
        json.dump(train_outputs,f)


    
    