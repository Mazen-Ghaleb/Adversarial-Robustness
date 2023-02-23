import cv2
import os
from pycocotools.coco import COCO
import numpy as np
from rich.progress import track
import shutil
import pandas as pd


def extract_signs(data_root_path, file_name, split_name):

    coco = COCO(os.path.join(data_root_path,
                "tsinghua_gtsdb_speedlimit", 'annotations', file_name))
    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(ids=img_ids)
    i = 0
    try:
        shutil.rmtree(os.path.join(data_root_path, "signs" ,split_name))
    except:
        pass

    os.makedirs(os.path.join(data_root_path, "signs",split_name))

    labels_dic = {"image_name": [], "src_image_name": [],"label": []}
    for img in track(imgs, f"Getting {split_name} sings"):
        img_array = cv2.imread(os.path.join(
            data_root_path, "tsinghua_gtsdb_speedlimit", f"{split_name}2017", img["file_name"]))
        annotation_ids = coco.getAnnIds(imgIds=img["id"])
        annotations = coco.loadAnns(ids=annotation_ids)
        src_image_name = img["file_name"]
        for annotation in annotations:
            x, y, w, h = np.asarray(annotation["bbox"], dtype=np.int32)
            if x < 0 or y < 0:
                continue
            bbox_img = img_array[y: y + h, x: x + w, :]
            bbox_img = cv2.resize(bbox_img, (64, 64))
            image_name = f"{i}.jpg"
            cv2.imwrite(os.path.join(data_root_path, "signs",
                        split_name, image_name), bbox_img)
            labels_dic["image_name"].append(image_name)
            labels_dic["src_image_name"].append(src_image_name)
            labels_dic["label"].append(annotation["category_id"])
            i += 1
    df = pd.DataFrame(labels_dic)
    df.to_csv(os.path.join(data_root_path, "signs", f"{split_name}.csv"), index=False)
        



if __name__ == "__main__":
    data_root_path = "datasets"
    extract_signs(data_root_path, "train2017.json", "train")
    extract_signs(data_root_path, "test2017.json", "test")
    extract_signs(data_root_path, "val2017.json", "val")
