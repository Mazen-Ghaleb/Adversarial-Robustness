from generate_attacked_samples import COCODataset
from model.speed_limit_detector import get_model
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import shutil
import h5py
import hdf5plugin

@torch.no_grad()
def dump_split_features(model, data_loader,split, datasets_path):
    with h5py.File(os.path.join(datasets_path, 'model_features', f'{split}.h5'),
                    'w') as hf:
        for (inputs,targets) in tqdm(data_loader,
                                    desc=f"Dumping features for {split}"):
            inputs = inputs.to(device)
            features = model(inputs, return_stems=True)
            features = [feature.cpu().squeeze(0).numpy() for feature in features]

            # Create a group for each item in the dataset
            group = hf.create_group(f'{targets[0].split(".")[0]}')
            
            group.create_dataset('p3', data=features[0],
                     **hdf5plugin.Zstd(clevel=22),
                     dtype=np.float32, shape=features[0].shape, chunks=features[0].shape)
                     
            group.create_dataset('p4', data=features[1],
                                **hdf5plugin.Zstd(clevel=22),
                                dtype=np.float32, shape=features[1].shape, chunks=features[1].shape)
                                
            group.create_dataset('p5', data=features[2],
                                **hdf5plugin.Zstd(clevel=22),
                                dtype=np.float32, shape=features[2].shape, chunks=features[2].shape)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    model.eval()
    datasets_path = os.path.join(os.path.dirname(os.getcwd()), 'model', 'datasets')

    try:
        shutil.rmtree(os.path.join(datasets_path, 'model_features', 'train'))
        shutil.rmtree(os.path.join(datasets_path, 'model_features', 'val'))
        shutil.rmtree(os.path.join(datasets_path, 'model_features', 'test'))
    except:
        pass

    os.makedirs(os.path.join(datasets_path, 'model_features', 'train'))
    os.makedirs(os.path.join(datasets_path, 'model_features', 'val'))
    os.makedirs(os.path.join(datasets_path, 'model_features', 'test'))

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

    # print("Dumping test output")
    
    # for i, (inputs,targets) in enumerate(tqdm(test_dataloader)):
    #     with torch.no_grad():
    #         inputs = inputs.to(device)
    #         outputs = model(inputs)[0].cpu().numpy()
    #         f = os.path.join(datasets_path, 'model_outputs',
    #                                 'test', f"{targets[0].split('.')[0]}.npy")
    #         np.save(f, outputs)
    
    # print("Dumping val output")

    # for i, (inputs,targets) in enumerate(tqdm(val_dataloader)):
    #     with torch.no_grad():
    #         inputs = inputs.to(device)
    #         outputs = model(inputs)[0].cpu().numpy()
    #         f = os.path.join(datasets_path, 'model_outputs',
    #                                 'val', f"{targets[0].split('.')[0]}.npy")
    #         np.save(f, outputs)


    # print("Dumping train output")
    
    # for i, (inputs,targets) in enumerate(tqdm(train_dataloader)):
    #     with torch.no_grad():
    #         inputs = inputs.to(device)
    #         features = model(inputs)[0]
    #         features = [feature.cpu().numpy() for feature in features]
    #         f = os.path.join(datasets_path, 'model_outputs',
    #                                 'train', f"{targets[0].split('.')[0]}.npy")
    #         np.save(f, outputs)

    dump_split_features(model, train_dataloader, "train", datasets_path)
    dump_split_features(model, test_dataloader, "test", datasets_path)
    dump_split_features(model, val_dataloader, "val", datasets_path)

    
    