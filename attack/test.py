from yolox_model import YoloxModel
from helper_functions import preprocess, read_img
from fast_attacks import fgsm, it_fgsm
from Inference import detect_sign
model_path = '../Carla Scripts/yolox_s.onnx'
model = YoloxModel(model_path, obj_threshold=0.6, cls_threshold=0.8)

img = read_img('./test_imgs/test.png')
preprocessed_img, r = preprocess(img)
print(detect_sign(preprocessed_img, model_path, r))
# preprocessed_img, r = preprocess(img)
# perturbed_img = it_fgsm(model, preprocessed_img, eps=4).numpy()
# print(detect_sign(perturbed_img, model_path, r))

# model = YoloxModel(model_path, obj_threshold=0.8, cls_threshold=0.2)
# perturbed_img = fgsm(model, preprocessed_img, eps=4).numpy()
# print(detect_sign(perturbed_img, model_path, r))




