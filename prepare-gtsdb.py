from pathlib import Path
from typing import Dict, List
from glob import glob
import numpy as np
import shutil
import os





def make_dir(dir: str) -> None:
    if not os.path.exists(dir):
        os.makedirs(dir)

# don't know what does this thing do for now
def xyxy_to_xyhx(x:List[int], w=1360, h=800) -> List[float]:
    y =  np.copy(x)
    # x cneter 
    y[0] = ((x[0] + x[2]) / 2) / w
    # y center
    y[1] = ((x[1] + x[3]) / 2) / h
    # width
    y[2] = (x[2] - x[0]) / w
    # hight
    y[3] = (x[3] - x[1]) / h
    return y


def save_label(file_name, lines):
    with open(file_name, 'w') as f:
        for line in lines:
            f.write(f'{line[0]} {line[1]} {line[2]} {line[3]} {line[4]}\n')


def get_annotations(gt_list:List[str], labels:List[str]=None) -> Dict:
    annotations = {}
    for line in gt_list:
        if line == "":
            break
        cols  = line.split(";")
        label = cols[-1]
        if labels is not None and not label in labels:
            continue

        id = cols[0]
        box = [float(x) for x in cols[1:len(cols)]]
        box = xyxy_to_xyhx(box)
        if id in annotations:
            annotations[id].append([label, *box])
        else:
            annotations[id] = [[label, *box]]
    return annotations

def prepare_full_data(dataroot:str, annotations:Dict) ->None:

    split = "full"
    labels_root = os.path.join(dataroot, split, 'labels')
    images_root = os.path.join(dataroot, split, 'images') 

    make_dir(labels_root)
    make_dir(images_root)

    for id in  annotations:
        label_path = os.path.join(labels_root, f'{Path(id).stem}.txt')
        image_path = os.path.join(dataroot, 'FullIJCNN2013', id)
        shutil.copy(image_path, images_root)
        save_label(label_path, annotations[id])

    return images_root, labels_root


def get_train_test_split(x, test_size:float, random_state:int=42):
    if test_size < 0.0 or test_size  > 0.99:
        print('error')
        return
    l = x.shape[0]
    indexes_list = [i for i in range(l)]
    np.random.shuffle(indexes_list)
    split_index = int(np.floor((1 - test_size) * l))
    train_indexes = indexes_list[:split_index]
    test_indexes = indexes_list[split_index:]
    return x[train_indexes], x[test_indexes]
    
    
        
def prepare_split(images, dataroot:str,split_name:str) ->None:
    images_root = os.path.join(dataroot, split_name, 'images')
    labels_root = os.path.join(dataroot, split_name, 'labels')
    make_dir(images_root)
    make_dir(labels_root)
    for image in images:
        p = Path(image)
        src_label = f'{dataroot}/full/labels/{p.stem}.txt'
        shutil.copy(image, images_root)
        try:
            shutil.copy(src_label, labels_root)
        except Exception as e:
            pass
    

def prepare():
    speed_limit_labels = ['0', '1', '2', '3', '4', '5', '7', '8']
    dataroot = 'datasets'
    gt_path = os.path.join(dataroot, 'FullIJCNN2013', 'gt.txt')
    gt_list = open(gt_path).read().split('\n')
    annotations = get_annotations(gt_list, speed_limit_labels)
    images_root, _ = prepare_full_data(dataroot, annotations)


    full_imglist = glob(f'{images_root}/*.ppm')
    full_imglist.sort()
    full_imglist = np.asarray(full_imglist)
    train, test = get_train_test_split(full_imglist, 0.3)
    test, validation = get_train_test_split(test, 0.5)

    prepare_split(train, dataroot, 'train')
    prepare_split(test, dataroot, 'test')
    prepare_split(validation,dataroot ,'validation')