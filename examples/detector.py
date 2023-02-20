from model.speed_limit_detector import SpeedLimitDetector
from glob import  glob
import cv2
import numpy as np
import os 
import torch



if torch.cuda.is_available():
    print("Running on cuda")
    device = torch.device("cuda")
else:
    print("Running on cpu")
    device = torch.device("cpu")

detector = SpeedLimitDetector(device)

test_imgs_path = os.path.join(os.path.dirname(__file__), "../test_imgs")
test_imgs_path = os.path.relpath(test_imgs_path, os.getcwd())
img = cv2.imread(os.path.join(test_imgs_path, "test.png"))
img = detector.preprocess(img)

imgs = np.asarray(img[None, :, :, :])
outputs = detector.get_model_output(imgs)
print(np.shape(outputs))
# for one image only use 
print(detector.decode_model_output(outputs[0]))

# for multiple images use:
# for output in outputs: 
#     print(detector.decode_model_output(output))