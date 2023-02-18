import json
import argparse
import copy
from typing import List, Dict
from rich.progress import track
import random
import os
import shutil

total = None
result_train = None
result_validation = None
result_test = None


def init_dic():
    global result_train, result_validation, result_test, total

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
                  "40", "15", "50", "60", "70", "80"]
    # 0 -> 100
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

    result_test = copy.deepcopy(total)
    result_validation = copy.deepcopy(total)
    result_train = copy.deepcopy(total)


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
    file = open(file_name, 'r').read()
    return json.loads(file)


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
    for annotation in track(data):
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
                xmin,
                ymin,
                xmax - xmin,
                ymax - ymin
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

    for img in track(data['imgs']):
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
    random.seed(42)
    random.shuffle(imgs)

    # Create a mapping from image ID to index
    image_id_to_index = {image['id'] : index for index, image in enumerate(imgs)}

    # Shuffle the annotations based on the shuffled images
    annotations.sort(key=lambda x: image_id_to_index[x['image_id']])


    # Create a mapping from image ID to annotations
    image_id_to_annotations = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(annotation)


    # uncomment this part if you need only the first 100 image for testing purposes 
    # Select the first 100 images
    # imgs = imgs[:100]

    # Get the corresponding annotations for the selected images
    # selected_annotations = []
    # for image in imgs:
    #     selected_annotations.extend(image_id_to_annotations[image['id']])
    # annotations = selected_annotations

    # Determine the partition sizes
    num_images = len(imgs)
    num_train = int(num_images * train_size)
    num_val = int(num_images * ((1 - train_size) / 2))
    num_test = num_images - num_train - num_val

    # Partition the images into test, train, and validation sets
    train_images = imgs[:num_train]
    test_images = imgs[num_train:num_train + num_test]
    val_images = imgs[num_train + num_test:]

    # Create the destination directories
    try:
        shutil.rmtree(os.path.join(data_root_path, 'tsinghua_gtsdb_speedlimit'))
    except:
        pass
    os.makedirs(os.path.join(data_root_path,'tsinghua_gtsdb_speedlimit', 'train2017'), exist_ok=True)
    os.makedirs(os.path.join(data_root_path,'tsinghua_gtsdb_speedlimit', 'test2017'), exist_ok=True)
    os.makedirs(os.path.join(data_root_path,'tsinghua_gtsdb_speedlimit', 'val2017'), exist_ok=True)
    os.makedirs(os.path.join(data_root_path,'tsinghua_gtsdb_speedlimit', 'annotations'), exist_ok=True)

    # Partition the annotations into test, train, and validation sets
    test_annotations = []
    val_annotations = []
    train_annotations = []

    print('copying test images')
    for image in track(test_images):
        test_annotations.extend(image_id_to_annotations[image['id']])
        src_path = os.path.join(data_path, image['file_name'])
        dest_path = os.path.join(data_root_path, 'tsinghua_gtsdb_speedlimit', 'test2017', image['file_name'])
        shutil.copy2(src_path, dest_path)

    print('copying validation images')
    for image in track(val_images):
        val_annotations.extend(image_id_to_annotations[image['id']])
        src_path = os.path.join(data_path, image['file_name'])
        dest_path = os.path.join(data_root_path, 'tsinghua_gtsdb_speedlimit', 'val2017', image['file_name'])
        shutil.copy2(src_path, dest_path)

    print('copying train images')
    for image in track(train_images):
        train_annotations.extend(image_id_to_annotations[image['id']])
        src_path = os.path.join(data_path, image['file_name'])
        dest_path = os.path.join(data_root_path, 'tsinghua_gtsdb_speedlimit', 'train2017', image['file_name'])
        shutil.copy2(src_path, dest_path)


    result_train['images'] = train_images
    result_train['annotations'] = train_annotations

    result_test['images'] = test_images
    result_test['annotations'] = test_annotations

    result_validation['images'] = val_images
    result_validation['annotations'] = val_annotations

    # Save the partitioned data
    with open(os.path.join(data_root_path,'tsinghua_gtsdb_speedlimit','annotations','test2017.json'), 'w') as f:
        json.dump(result_test, f)
    with open(os.path.join(data_root_path,'tsinghua_gtsdb_speedlimit', 'annotations','val2017.json'), 'w') as f:
        json.dump(result_validation, f)
    with open(os.path.join(data_root_path,'tsinghua_gtsdb_speedlimit', 'annotations','train2017.json'), 'w') as f:
        json.dump(result_train, f)



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

    print("Parsing tsinghua dataset")
    parse_tsinghua(tsinghua_data)

    gtsdb_labels_to_keep = {'0': '20', '1': '30', '2': '50', '3': '60',
                            '4': '70', '5': '80', '7': '100', '8': '120'}
    gtsdb_labels_to_keep = ['0', '1', '2', '3', '4', '5', '7', '8']

    gtsdb_data = load_txt(gtsdb_data_path, gtsdb_labels_to_keep)
    print("Parsing gtsdb dataset")
    parse_gtsdb(gtsdb_data)

    shuffle_and_split(total)

    print('total Images: ' + str(len(total['images'])))
    print('total Annotations: ' + str(len(total['annotations'])))

    print('Train Images: ' + str(len(result_train['images'])))
    print('Train Annotations: ' + str(len(result_train['annotations'])))

    print('validation Images: ' + str(len(result_validation['images'])))
    print('validation Annotations: ' +
          str(len(result_validation['annotations'])))

    print('Test Images: ' + str(len(result_test['images'])))
    print('Test Annotations: ' + str(len(result_test['annotations'])))

