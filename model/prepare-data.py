import json
import argparse
import copy
from typing import List, Dict
from rich.progress import track
import random
import os
import shutil
from pycocotools.coco import COCO
from collections import defaultdict
import numpy as np
import cv2

total = None
result_train = None
result_validation = None
result_test = None


def init_dic():
    global  total

    total = {
        "info": {
            "description": "TT100K Dataset (ver. 2021) COCO Format",
            "url": "https://github.com/zhaoweizhong",
            "version": "2.0",
            "year": 2021,
            "contributor": "Zhaowei Zhong",
            "date_created": "2021/03/05"
        },
        "licenses": [
            {
                "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    categories = ["100", "120", "20",  "30",
                  "40", "15", "50", "60", "70", "80"] # 0 -> 100
    # 1 -> 120
    # 2 -> 20
    # 3 -> 30
    # 4 -> 40
    # 5 -> 15
    # 7 -> 60
    # 8 -> 70
    # 9 -> 80
    for i, category in enumerate(categories):
        total['categories'].append({
            "id": i,
            "name": category
        })


def load_txt(file_name, labels_to_keep):
    file = open(file_name, 'r')
    data = []
    for line in file.readlines():
        line = line.replace('\n', '')
        if line.split(";")[-1] not in labels_to_keep:
            continue
        data.append(line)
    return data


def load_json(file_name):
    with open(file_name, 'r') as file:
        # file = open(file_name, 'r').read()
        return json.load(file)


def parse_gtsdb(data):
    global total

    # keys for the following dic
    # 0 -> 100
    # 1 -> 120
    # 2 -> 20
    # 3 -> 30
    # 4 -> 40
    # 5 -> 15
    # 7 -> 60
    # 8 -> 70
    # 9 -> 80
    categories_dic = {"7": 0, "8": 1,
                      "0": 2,  "1": 3, "2": 6, "3": 7, "4": 8, "5": 9}
    for annotation in track(data, "parsing gtsdb"):
        s = annotation.split(';')
        img_id = int(s[0][:5])
        img_name = s[0][:]
        xmin = int(s[1])
        ymin = int(s[2])
        xmax = int(s[3])
        ymax = int(s[4])
        class_id = categories_dic[s[5]]
        anno_id = len(total['annotations'])
        if not bool([True for img in total['images'] if img['id'] == img_id]):
            total['images'].append({
                "license": 1,
                "file_name": img_name,
                "height": 800,
                "width": 1360,
                "id": img_id
            })
        total['annotations'].append({
            "segmentation": [[]],
            "area": (xmax - xmin) * (ymax - ymin),
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": [
                int(xmin),
                int(ymin),
                int(xmax - xmin),
                int(ymax - ymin)
            ],
            "category_id": class_id,
            "id": anno_id
        })


def parse_tsinghua(data):
    global result_train, result_validation, result_test

    # Categories
    # categories = ["pl80", "pl90", "pl60",  "pl15", "pl70", "pl5", "pl30","pl20","pl40","pl120","pl10","pl100", "pl110","pl35", "pl25", "pl50", "pl65"]
    categories_dic = {"pl100": 0, "pl120": 1,
                      "pl20": 2,  "pl30": 3, "pl40": 4, "pl15": 5, "pl50": 6, "pl60": 7, "pl70": 8, "pl80": 9}
    categories = ["pl100", "pl120", "pl20",  "pl30",
                  "pl40", "pl15", "pl50", "pl60", "pl70", "pl80"]
    # # 0 -> 100
    # # 1 -> 120
    # # 2 -> 20
    # # 3 -> 30
    # # 4 -> 40
    # # 5 -> 15
    # # 7 -> 60
    # # 8 -> 70
    # # 9 -> 80

    # Images and Annotations

    for img in track(data['imgs'], "parsing tsinghua"):
        flag = False
        for box in data['imgs'][img]['objects']:
            if box['category'] in categories:
                flag = True
        if flag:
            total['images'].append({
                "license": 1,
                "file_name": data['imgs'][img]['path'].split('/')[1],
                "height": 2048,
                "width": 2048,
                "id": data['imgs'][img]['id']
            })
            for box in data['imgs'][img]['objects']:
                anno_id = len(total['annotations'])
                if box['category'] in categories:
                    total['annotations'].append({
                        "segmentation": [[]],
                        "area": (box['bbox']['xmax'] - box['bbox']['xmin']) * (box['bbox']['ymax'] - box['bbox']['ymin']),
                        "iscrowd": 0,
                        "image_id": data['imgs'][img]['id'],
                        "bbox": [
                            box['bbox']['xmin'],
                            box['bbox']['ymin'],
                            box['bbox']['xmax'] - box['bbox']['xmin'],
                            box['bbox']['ymax'] - box['bbox']['ymin']
                        ],
                        "category_id": categories_dic[box['category']],
                        "id": anno_id
                    })
                    if ('ellipse_org' in box):
                        for xy in box['ellipse_org']:
                            total['annotations'][-1]['segmentation'][0].append(
                                xy[0])
                            total['annotations'][-1]['segmentation'][0].append(
                                xy[1])
                    elif 'polygon' in box:
                        for xy in box['polygon']:
                            total['annotations'][-1]['segmentation'][0].append(
                                xy[0])
                            total['annotations'][-1]['segmentation'][0].append(
                                xy[1])

    # with open('test.json', "w") as f:
    #     json.dump(result_test, f)
"""
filters the annotations to only keep the images that contain any of the passed labels
"""


def filter_annotations(annotations: Dict, labels_to_keep: List[str]) -> Dict:
    imgs_to_keep = {}
    imgs = annotations['imgs']
    for img_id in imgs:
        objects = imgs[img_id]['objects']
        for obj in objects:
            if obj['category'] in labels_to_keep:
                if img_id not in imgs_to_keep:
                    imgs_to_keep[img_id] = imgs[img_id]
                    imgs_to_keep[img_id]['objects'] = []
                    imgs_to_keep[img_id]['objects'].append(obj)
    annotations['imgs'] = imgs_to_keep
    return annotations


def shuffle_and_split(data, data_path="datasets/tsinghua_gtsdb_full",
                      data_root_path="datasets", train_size=0.7):

    annotations = data['annotations']
    imgs = data['images']

    # Shuffle the list of images
    np.random.seed(42)
    np.random.shuffle(imgs)

    # Create a mapping from image ID to index
    image_id_to_index = {image['id']
        : index for index, image in enumerate(imgs)}

    # Shuffle the annotations based on the shuffled images
    annotations.sort(key=lambda x: image_id_to_index[x['image_id']])

    # Create a mapping from image ID to annotations
    image_id_to_annotations = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(annotation)

    category_to_annotations = defaultdict(list)
    for annotation in annotations:
        category_to_annotations[annotation['category_id']].append(annotation)

    category_sizes = {category_id: len(
        annotation) for category_id, annotation in category_to_annotations.items()}

    category_train_sizes = {category_id: int(
        size * train_size) for category_id, size in category_sizes.items()}
    category_val_sizes = {category_id: int(
        size * ((1 - train_size) / 2)) for category_id, size in category_sizes.items()}

    test_annotations = []
    val_annotations = []
    train_annotations = []
    for category_id, annotations in category_to_annotations.items():
            random.shuffle(annotations)
            train_annotations.extend(annotations[:category_train_sizes[category_id]])
            val_annotations.extend(annotations[category_train_sizes[category_id]:category_train_sizes[category_id]+category_val_sizes[category_id]])
            test_annotations.extend(annotations[category_train_sizes[category_id]+category_val_sizes[category_id]:])

    # Get the corresponding images for the selected annotations
    selected_image_ids = set(annotation['image_id'] for annotation in train_annotations + val_annotations + test_annotations)

    selected_images = [image for image in imgs if image['id'] in selected_image_ids]



    # uncomment this part if you need only the first 100 image for testing purposes
    # Select the first 100 images
    # imgs = imgs[:100]

    # Get the corresponding annotations for the selected images
    # selected_annotations = []
    # for image in imgs:
    #     selected_annotations.extend(image_id_to_annotations[image['id']])
    # annotations = selected_annotations

    image_id_to_index = {image['id']: image for  image in selected_images}

    try:
        shutil.rmtree(os.path.join(data_root_path, "tsinghua_gtsdb_speedlimit"))
    except:
        pass
    os.makedirs(os.path.join(data_root_path,'tsinghua_gtsdb_speedlimit'), exist_ok=True)
    os.makedirs(os.path.join(data_root_path,'tsinghua_gtsdb_speedlimit', 'train2017'), exist_ok=True)
    os.makedirs(os.path.join(data_root_path,'tsinghua_gtsdb_speedlimit', 'test2017'), exist_ok=True)
    os.makedirs(os.path.join(data_root_path,'tsinghua_gtsdb_speedlimit', 'val2017'), exist_ok=True)
    os.makedirs(os.path.join(data_root_path,'tsinghua_gtsdb_speedlimit', 'annotations'),exist_ok=True)


    done = set()
    train_images = []
    test_images = []
    val_iamges = []
    for annotation in track(train_annotations, "copying train images"):    
        image = image_id_to_index[annotation['image_id']]
        image_name = image['file_name']
        if image_name in done:
            continue
        train_images.append(image)
        done.add(image_name)
        dest_path = os.path.join(data_root_path, 'tsinghua_gtsdb_speedlimit', 'train2017', image_name)
        image_path = os.path.join(data_root_path, 'tsinghua_gtsdb_full', image_name)
        shutil.copy(image_path, dest_path)

    for annotation in track(test_annotations, "copying test images"):    
        image = image_id_to_index[annotation['image_id']]
        image_name = image['file_name']
        if image_name in done:
            continue
        test_images.append(image)
        done.add(image_name)
        dest_path = os.path.join(data_root_path, 'tsinghua_gtsdb_speedlimit', 'test2017', image_name)
        image_path = os.path.join(data_root_path, 'tsinghua_gtsdb_full', image_name)
        shutil.copy(image_path, dest_path)

    for annotation in track(val_annotations, "copying validation images"):    
        image = image_id_to_index[annotation['image_id']]
        image_name = image['file_name']
        if image_name in done:
            continue
        val_iamges.append(image)
        done.add(image_name)
        dest_path = os.path.join(data_root_path, 'tsinghua_gtsdb_speedlimit', 'val2017', image_name)
        image_path = os.path.join(data_root_path, 'tsinghua_gtsdb_full', image_name)
        shutil.copy(image_path, dest_path)
    
        

    # Create the annotation files for train, validation, and test sets
    train_data = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": data["categories"]
    }
    with open(os.path.join(data_root_path, 'tsinghua_gtsdb_speedlimit', 'annotations', 'train2017.json'), 'w') as f:
        json.dump(train_data, f)

    val_data = {
        "images": val_iamges,
        "annotations": val_annotations,
        "categories": data["categories"]
    }
    with open(os.path.join(data_root_path, 'tsinghua_gtsdb_speedlimit', 'annotations', 'val2017.json'), 'w') as f:
        json.dump(val_data, f)

    test_data = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": data["categories"]
    }
    with open(os.path.join(data_root_path, 'tsinghua_gtsdb_speedlimit', 'annotations', 'test2017.json'), 'w') as f:
        json.dump(test_data, f)


    

def oversample(data_root_path, file_name):
    np.random.seed(42)
    coco_annotations_file = os.path.join(data_root_path, 'tsinghua_gtsdb_speedlimit', 'annotations',
                                         file_name)
    # Load the COCO dataset
    coco = COCO(coco_annotations_file)

    # Get a list of all image IDs in the dataset
    image_ids = coco.getImgIds()

    # Group image IDs by class
    class_to_images = defaultdict(list)
    for image_id in image_ids:
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        for annotation in annotations:
            class_id = annotation["category_id"]
            class_to_images[class_id].append(image_id)

    # Compute the number of images to oversample for each class
    # max_cls_count = 400
    class_counts = defaultdict(int)
    for class_id, images in class_to_images.items():
        # images_ = images[:max_cls_count]
        # class_to_images[class_id] = images_
        class_counts[class_id] = len(images)
        # max_cls_count = max(max_cls_count, class_counts[class_id])

    oversample_counts = defaultdict(int)

    print(class_counts)
    for class_id, class_count in class_counts.items():
        if class_count < max_cls_count:
            oversample_counts[class_id] = max_cls_count - class_count
        else:
            oversample_counts[class_id] = 0

    print("oversample_counts", oversample_counts)

    # Randomly sample images to oversample for each class
    oversample_image_ids = []
    for class_id, count in oversample_counts.items():
        if count == 0:
            images_to_oversample = class_to_images[class_id]
            print(len(images_to_oversample))
        else:
            images_to_oversample = map(lambda x: int(
                x), np.random.choice(class_to_images[class_id], count))
        oversample_image_ids.extend(images_to_oversample)

    # Create a new COCO dataset with the oversampled images
    oversampled_dataset = {"images": [], "annotations": [],
                           "categories": coco.dataset["categories"]}
    for i, image_id in enumerate(oversample_image_ids):
        image_info = coco.imgs[image_id]
        new_image_info = {"file_name": image_info["file_name"], "height": image_info["height"],
                          "width": image_info["width"], "id": i}
        oversampled_dataset["images"].append(new_image_info)
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        print("len", len(annotations))
        for annotation in annotations:
            new_annotation = dict(annotation)
            new_annotation["id"] = len(oversampled_dataset["annotations"])
            new_annotation["image_id"] = i
            oversampled_dataset["annotations"].append(new_annotation)

    len(oversampled_dataset["images"])
    # Write the oversampled dataset to a new annotations file
    # output_file = os.path.join(coco_annotations_file, "train2017.json")
    with open(coco_annotations_file, "w") as f:
        json.dump(oversampled_dataset, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str,
                        default='datasets/tsinghua and gtsdb')

    args = parser.parse_args()
    root = args.root
    gtsdb_data_path = root + '/gt.txt'
    tsinghua_data_path = root + '/annotations.json'
    init_dic()
    tsinghua_data = load_json(tsinghua_data_path)
    tsinghua_labels_to_keep = ["pl100", "pl120", "pl20",  "pl30",
                               "pl40", "pl15", "pl50", "pl60", "pl70", "pl80"]
    tsinghua_data = filter_annotations(tsinghua_data, tsinghua_labels_to_keep)

    parse_tsinghua(tsinghua_data)

    gtsdb_labels_to_keep = {'0': '20', '1': '30', '2': '50', '3': '60',
                            '4': '70', '5': '80', '7': '100', '8': '120'}
    gtsdb_labels_to_keep = ['0', '1', '2', '3', '4', '5', '7', '8']

    gtsdb_data = load_txt(gtsdb_data_path, gtsdb_labels_to_keep)

    parse_gtsdb(gtsdb_data)

    shuffle_and_split(total)

    # oversample("datasets", "train2017.json")
    # oversample("datasets", "test2017.json")
    # oversample("datasets", "val2017.json")
    print('total Images: ' + str(len(total['images'])))
    print('total Annotations: ' + str(len(total['annotations'])))

    print('Train Images: ' + str(len(result_train['images'])))
    print('Train Annotations: ' + str(len(result_train['annotations'])))

    print('validation Images: ' + str(len(result_validation['images'])))
    print('validation Annotations: ' +
          str(len(result_validation['annotations'])))

    print('Test Images: ' + str(len(result_test['images'])))
    print('Test Annotations: ' + str(len(result_test['annotations'])))

    coco_annotations_file = "datasets/tsinghua_gtsdb_speedlimit/annotations/train2017.json"
    coco = COCO(coco_annotations_file)
    print(len(coco.getImgIds()))
