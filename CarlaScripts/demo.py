from model.speed_limit_detector import SpeedLimitDetector
from model.sign_classifier import SignClassifier
from attack.fgsm import FGSM
from attack.iterative_fgsm import ItFGSM
import torch
import numpy as np
from attack.attack_base import AttackBase
import torch.nn.functional as F
from model.sign_classifier import classifier_loss, classifier_target_generator
from model.custom_yolo import yolox_loss, yolox_target_generator
from defense.high_level_guided_denoiser import get_HGD_model

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
        self.attacks = {"FGSM": FGSM(), "IT-FGSM": ItFGSM()}
        self.defenses = {"HGD": get_HGD_model(self.device)}

    def __crop_signs(self, detection_boxes):
        delete_masks = []
        cropped_signs = []
        for i, box in enumerate(np.array(detection_boxes, dtype=np.int32)):
            if np.any(box < 0, axis=None):
                delete_masks.append(i)
                continue
            xmin, ymin, xmax, ymax = box
            cropped_sign = self.image[ymin: ymax, xmin:xmax, :]
            if cropped_sign.size == 0:
                delete_masks.append(i)
                continue
            cropped_sign = self.classifier.preprocess(cropped_sign)
            cropped_signs.append(cropped_sign)
        
        cropped_signs = np.asarray(cropped_signs)
        detection_boxes = np.delete(detection_boxes, delete_masks, axis=0)
        return cropped_signs, detection_boxes

    
    def preprocess(self, image:np.ndarray):
        self.image = image
        self.preprocessed_image = self.detector.preprocess(image)

    def run_without_attack(self, debug=False):
        images = torch.from_numpy(self.preprocessed_image[None, :, :, :]).to(self.device)
        detection_output = self.detector.get_model_output(images)[0]
        detection_output = self.detector.decode_model_output(detection_output)

        if detection_output is None:
            return None

        detection_label, detection_conf, detection_boxes = detection_output
        if debug:
            print(detection_label, detection_conf)

        cropped_signs, detection_boxes = self.__crop_signs(detection_boxes)
        cropped_signs = torch.from_numpy(cropped_signs).float().to(self.device)
        classification_labels, classification_conf =  self.classifier.classify_signs(cropped_signs)

        return self.classes[classification_labels], classification_conf, detection_boxes
    
    def run_with_attack(self, attack_type, debug=False):

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
        if debug:
            print(detection_label, detection_conf)

        cropped_signs, detection_boxes = self.__crop_signs(detection_boxes)
        cropped_signs = torch.from_numpy(cropped_signs).float().to(self.device)


        attack.model = self.classifier.model
        attack.loss = classifier_loss
        attack.target_generator = classifier_target_generator
        perturbed_cropped_signs = attack.generate_attack(cropped_signs)

        classification_labels, classification_conf =  self.classifier.classify_signs(perturbed_cropped_signs)
        return self.classes[classification_labels], classification_conf, detection_boxes

    def run_with_defense(self, defense_type, attack_type):
        defense_model = self.defenses[defense_type]
        
        attack: AttackBase = self.attacks[attack_type]
        images = torch.from_numpy(self.preprocessed_image[None, :, :, :]).to(self.device)

        attack.model = self.detector.model
        attack.loss = yolox_loss
        attack.target_generator = yolox_target_generator
        perturbed_images = attack.generate_attack(images)
        
        defense_model.eval()
        with torch.no_grad():
            denoised_images = perturbed_images - defense_model(perturbed_images)

        detection_output = self.detector.get_model_output(denoised_images)[0]
        detection_output = self.detector.decode_model_output(detection_output)
        
        if detection_output is None:
            return None
        else:
            # classification_labels, classification_conf, detection_boxes =  detection_output
            return detection_output

if __name__ == "__main__":
    import os
    import cv2
    test_imgs_path = os.path.join(os.path.dirname(__file__), "../test_imgs")
    test_imgs_path = os.path.relpath(test_imgs_path, os.getcwd())
    img = cv2.imread(os.path.join(test_imgs_path, "test.png"))
    demo = Demo()
    demo.preprocess(img)
    output = demo.run_without_attack()
    print("without attack: ")
    print(output[0], output[1], output[2])
    # output = demo.run_with_defense("HGD","FGSM")
    # print(output[0], output[1], output[2])
    print("with attack: ")
    output = demo.run_with_attack("IT-FGSM")
    print(output[0], output[1], output[2])

    # output = demo.run_with_attack("FGSM")
    # print("with fgsm attack: ")
    # print(output[0], output[1], output[2])

    # output = demo.run_with_attack("IT-FGSM")
    # print("with it-fgsm attack: ")
    # print(f"labels: \n{output[0]}")
    # print(f"conf: \n{output[1]}")
    # print(f"boxes: \n{output[2]}")