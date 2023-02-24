from model.speed_limit_detector import SpeedLimitDetector
from model.sign_classifier import SignClassifier
from attack.fgsm import FGSM
from attack.iterative_fgsm import ItFGSM
import torch
import numpy as np
from attack.attack_base import AttackBase
import torch.nn.functional as F

class Demo:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            print("Running on cuda")
            self.device = torch.device("cuda")
        else:
            print("Running on CPU")
            self.device = torch.device("cpu")
        self.detector = SpeedLimitDetector(self.device)
        self.classifier = SignClassifier(self.device)
        self.classes = np.array([100, 120, 20, 30, 40, 15, 50, 60, 70, 80])
        self.attacks = {"fgsm": FGSM(), "it-fgsm": ItFGSM()}
    
    def preprocess(self, image:np.ndarray):
        self.preprocessed_image = self.detector.preprocess(image)

    def run_without_attack(self,image:np.ndarray):
        images = torch.from_numpy(self.preprocessed_image[None, :, :, :]).to(self.device)
        detection_output = self.detector.get_model_output(images)[0]
        detection_output = self.detector.decode_model_output(detection_output)
        if detection_output is None:
            return None
        detection_label, detection_conf, detection_boxes = detection_output

        cropped_signs = []
        for box in np.asarray(detection_boxes, dtype=int):
            xmin, ymin, xmax, ymax = box
            cropped_sign = self.classifier.preprocess(image[ymin:ymax, xmin: xmax, :])
            cropped_signs.append(cropped_sign)
        cropped_signs = np.asarray(cropped_signs)
        
        cropped_signs = torch.from_numpy(cropped_signs).float().to(self.device)
        
        classification_labels, classification_conf =  self.classifier.classify_signs(cropped_signs)
        return self.classes[classification_labels], classification_conf, detection_boxes
    
    def run_with_attack(self, image:np.ndarray, attack_type):

        attack: AttackBase = self.attacks[attack_type]

        images = torch.from_numpy(self.preprocessed_image[None, :, :, :]).to(self.device)

        attack.model = self.detector.model
        attack.loss = yolox_loss
        attack.target_generator = yolox_target_generator
        perturbed_images = attack.generate_attack(images)

        detection_output = self.detector.get_model_output(perturbed_images)[0]
        detection_output = self.detector.decode_model_output(detection_output)

        if detection_output is None:
            return None
        detection_label, detection_conf, detection_boxes = detection_output

        cropped_signs = []
        for box in np.asarray(detection_boxes, dtype=int):
            xmin, ymin, xmax, ymax = box
            cropped_sign = self.classifier.preprocess(image[ymin:ymax, xmin: xmax, :])
            cropped_signs.append(cropped_sign)
        cropped_signs = np.asarray(cropped_signs)

        cropped_signs = torch.from_numpy(cropped_signs).float().to(self.device)

        attack.model = self.classifier.model
        attack.loss = F.cross_entropy
        attack.target_generator = classifier_target_generator
        perturbed_cropped_signs = attack.generate_attack(cropped_signs)

        classification_labels, classification_conf =  self.classifier.classify_signs(perturbed_cropped_signs)
        return self.classes[classification_labels], classification_conf, detection_boxes

def yolox_loss(outputs, targets):
    loss_cls = F.binary_cross_entropy(outputs[:, :, 5:], targets[:, :, 5:])
    loss_objs = F.binary_cross_entropy(outputs[:, :, 4], targets[:, :, 4])
    return loss_cls.sum() + loss_objs.sum()

def yolox_target_generator(outputs):
    obj_threshold  = 0.5
    cls_threshold  = 0.5
    with torch.no_grad():
        objs_targets = (outputs[:, :, 4] > obj_threshold).float().unsqueeze(dim=2)
        cls_targets = (outputs[:, :, 5:] > cls_threshold).float()
        return torch.cat((outputs[:, :, :4], objs_targets, cls_targets), dim=2)

def classifier_target_generator(outputs):
    labels = outputs.argmax(1)
    targets = F.one_hot(labels, 10).float()
    return targets

    



if __name__ == "__main__":
    import os
    import cv2
    test_imgs_path = os.path.join(os.path.dirname(__file__), "../test_imgs")
    test_imgs_path = os.path.relpath(test_imgs_path, os.getcwd())
    img = cv2.imread(os.path.join(test_imgs_path, "test.png"))
    demo = Demo()
    demo.preprocess(img)
    output = demo.run_without_attack(img)
    print("without attack: ")
    print(output[0], output[1], output[2])


    output = demo.run_with_attack(img, "fgsm")
    print("with fgsm attack: ")
    print(output[0], output[1], output[2])

    output = demo.run_with_attack(img, "it-fgsm")
    print("with it-fgsm attack: ")
    print(f"labels: \n{output[0]}")
    print(f"conf: \n{output[1]}")
    print(f"boxes: \n{output[2]}")