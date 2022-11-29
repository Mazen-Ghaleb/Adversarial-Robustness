from carla.Inference import OnnxModel
from attack import fgsm, YoloxModel, it_fgsm
import torch
import cv2

if __name__ == "__main__":
    # initialization of the model used for prediction should be done once
    model_path = "../carla/yolox_s.onnx"
    model = OnnxModel(model_path)

    # check if cuda is available
    cuda_available = torch.cuda.is_available()

    # initialization of the model used for attack
    # should  be done once
    # these thresholds are used for fgsm
    attack_model = YoloxModel(model_path, obj_threshold=0.8, cls_threshold=0.6)

    # check to use gpu
    if cuda_available:
        attack_model = attack_model.cuda()

    # load the image into memory
    img_path = "../test_imgs/test.png"
    img = cv2.imread(img_path)

    # preprocess the image
    preprocessed_img = model.preprocess(img)

    # the attack should be generated on the preprocessed image


    # fgsm
    # generated perturbed image
    perturbed_image = fgsm(attack_model, preprocessed_img, eps=4, cuda=cuda_available).numpy()
    print(perturbed_image.shape)
    prediction = model.detect_sign(perturbed_image)
    print(f"fgsm results: ")

    print(f"prediction is :{prediction[0]}")
    print(f"confidence is :{prediction[1]}")

    # it_fgsm
    # use these values for it_fgsm
    attack_model.set_thresholds(obj_thresholds=.9, cls_thersholds=0.6)
    perturbed_image = it_fgsm(attack_model, preprocessed_img, cuda=cuda_available).numpy()

    prediction = model.detect_sign(perturbed_image)
    print(f"it_fgsm results: ")

    print(f"prediction is :{prediction[0]}")
    print(f"confidence is :{prediction[1]}")
