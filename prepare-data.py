from logging import root
from operator import gt
from pathlib import Path
from re import I
import shutil
import sre_compile
from typing import List
from unicodedata import name
from xmlrpc.client import DateTime
import numpy as np
import os
from glob import glob



verbose = False

def make_dir(dir: Path) -> None:
    if not os.path.exists(dir):
        os.makedirs(dir)

# don't know what does this thing do for now
def xyxy_to_xyhx(x:List[int], w=1360, h=800) -> List[float]:
    y =  np.copy(x)
    y[0] = ((x[0] + x[2]) / 2) / w
    y[1] = ((x[1] + x[3]) / 2) / h
    y[2] = (x[2] - x[0]) / w
    y[3] = (x[3] - x[1]) / h
    return y


def save_label(file_name, lines):
    with open(file_name, 'w') as f:
        for line in lines:
            f.write(f'{line[0]} {line[1]} {line[2]} {line[3]} {line[4]}\n')


def get_target_dictionary(gt_list:List[str]):
    target_dictionary = {}
    for line in gt_list:
        if line == "":
            break
        cols  = line.split(";")
        file_name = cols[0]
        label = cols[-1]
        box = [float(x) for x in cols[1:len(cols)]]
        box = xyxy_to_xyhx(box)
        if file_name in target_dictionary:
            target_dictionary[file_name].append([label, *box])
        else:
            target_dictionary[file_name] = [[label, *box]]
    return target_dictionary
        

def main():
    print(os.getcwd())
    global verbose
    data_root = 'data'
    gt_path = os.path.join(data_root, 'FullIJCNN2013', 'gt.txt')
    gt_list = open(gt_path).read().split('\n')
    if verbose:
        print(gt_list[:10])

    target_dictionary = get_target_dictionary(gt_list)
    split = "full"

    if verbose:
        print(f'starting saving labels for ${split} dataset')

    labels_root = os.path.join(data_root, split, 'labels')
    make_dir(labels_root)
    for key in  target_dictionary:
        path = os.path.join(labels_root, f'{Path(key).stem}.txt')
        save_label(path, target_dictionary[key])

    if verbose:
        print(f'starting saving images for ${split} dataset')

    images_path = os.path.join(data_root, 'FullIJCNN2013')
    images_list = glob(f'{images_path}/*.ppm')
    images_root = os.path.join(data_root, split, 'images') 
    make_dir(images_root)

    for image in images_list:
        shutil.copy(image, images_root)

    full_imglist = glob(f'{images_root}/*.ppm')
    full_imglist.sort()

    #print(len(full_imglist)); full_imglist[:5]
    split = 'train'
    if verbose:
        print(f'preparing {split}')
    train_images_list = images_list[:600]
    images_root = os.path.join(data_root, split, 'images')
    labels_root = os.path.join(data_root, split, 'labels')
    make_dir(images_root)
    make_dir(labels_root)
    for image in train_images_list:
        p = Path(image)
        src_label = f'{data_root}/full/labels/{p.stem}.txt'
        shutil.copy(image, images_root)
        try:
            shutil.copy(src_label, labels_root)
        except Exception as e:
            pass
    

    split = 'validation'
    if verbose:
        print(f'preparing {split}')
    validation_images_list = images_list[600:]
    images_root = os.path.join(data_root, split, 'images')
    labels_root = os.path.join(data_root, split, 'labels')
    make_dir(images_root)
    make_dir(labels_root)
    for image in validation_images_list:
        p = Path(image)
        src_label = f'{data_root}/full/labels/{p.stem}.txt'
        shutil.copy(image, images_root)
        try:
            shutil.copy(src_label, labels_root)
        except Exception as e:
            pass

if __name__ == '__main__':
    main()