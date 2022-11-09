import numpy as np
import cv2

def read_img(img_path):
    img = cv2.imread(img_path)
    return img

def preprocess(img, input_size=[640, 640], swap=(2, 0, 1), pad=True):
    if not pad:
        resized =  cv2.resize(img, input_size)
        resized = resized.transpose(swap)
        return np.ascontiguousarray(resized, dtype=np.float32)
    if len(img.shape) == 3:
        padded_img = np.ones(
            (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0],
            input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r),
               : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    return padded_img