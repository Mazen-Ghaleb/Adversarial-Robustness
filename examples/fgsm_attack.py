from model.speed_limit_detector import SpeedLimitDetector
from attack import fgsm, it_fgsm
import torch
import cv2
import numpy as np

if torch.cuda.is_available():
    print("Running on CUDA")
    device = torch.device('cuda')
else:
    print("Running on cpu")
    device = torch.device('cpu')

if __name__ == "__main__":

    # initialization of the model used for prediction should be done once
    detector = SpeedLimitDetector(device)


    # load the image into memory
    img_path = "../test_imgs/test2.png"
    img = cv2.imread(img_path)
    # preprocess the image
    preprocessed_img = detector.preprocess_model_input(img)

    preprocessed_imgs = np.asarray(preprocessed_img[None, :, :, :])
    preprocessed_imgs = torch.from_numpy(preprocessed_imgs)


    # fgsm
    # generated perturbed images
    perturbed_images = fgsm(detector.model, preprocessed_imgs, device,eps=4, return_numpy=True)

    outputs = detector.get_model_output(perturbed_images)
    prediction = detector.decode_model_output(outputs[0])

    print(f"fgsm results: ")

    print(f"prediction is :{prediction[0]}")
    print(f"confidence is :{prediction[1]}")
    print(f"bounding box is :{prediction[2]}")

    # # it_fgsm
    perturbed_image = it_fgsm(detector.model, preprocessed_imgs, device, eps=4, return_numpy=True)
    outputs = detector.get_model_output(perturbed_images)
    prediction = detector.decode_model_output(outputs[0])

    print(f"it-fgsm results: ")

    print(f"prediction is :{prediction[0]}")
    print(f"confidence is :{prediction[1]}")
    print(f"bounding box is :{prediction[2]}")