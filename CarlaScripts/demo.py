from model.speed_limit_detector import SpeedLimitDetector

from attack.attack_base import AttackBase
from attack.iterative_fgsm import ItFGSM
from attack.fgsm import FGSM

import torch
import numpy as np
from model.custom_yolo import yolox_loss, yolox_target_generator
from defense.hgd_trainer import get_HGD_model
from timeit import default_timer as timer

class Demo:
    def __init__(self, confidence_threshold=0.8) -> None:
        if torch.cuda.is_available():
            print("Running on CUDA")
            self.device = torch.device("cuda")
        else:
            print("Running on CPU")
            self.device = torch.device("cpu")

        self.confidence_threshold = confidence_threshold
        self.detector = SpeedLimitDetector(self.device)
        self.attacks = {"FGSM": FGSM(yolox_target_generator, yolox_loss),
                        "IT-FGSM": ItFGSM(yolox_target_generator, yolox_loss)}
        self.defenses = {"HGD": get_HGD_model(self.device)}

    def __sort_labels(self, cls_labels, cls_confs, detection_boxes):
        sorted_indexes = np.argsort(cls_confs)[::-1]
        return cls_labels[sorted_indexes], cls_confs[sorted_indexes], detection_boxes[sorted_indexes]

    
    def preprocess(self, image:np.ndarray):
        self.image = image
        self.preprocessed_image = self.detector.preprocess(image)

    def run_without_attack(self, debug=False):
        start = timer()
        images = torch.from_numpy(self.preprocessed_image[None, :, :, :]).to(self.device)
        detection_output = self.detector.get_model_output(images)[0]
        detection_output = self.detector.decode_model_output(detection_output, self.confidence_threshold)
        total_time = timer() - start

        if detection_output is None:
            return None

        detection_label, detection_conf, detection_boxes = detection_output
        if debug:
            print(detection_label, detection_conf)
        return self.__sort_labels(detection_label, detection_conf, detection_boxes) + (total_time,)
    
    def run_with_attack(self, attack_type, debug=False):

        attack: AttackBase = self.attacks[attack_type]

        images = torch.from_numpy(self.preprocessed_image[None, :, :, :]).to(self.device)

        attack.model = self.detector.model
        attack.loss = yolox_loss
        start = timer()
        self.detector_attacked_images = perturbed_images

        detection_output = self.detector.get_model_output(perturbed_images)[0]
        detection_output = self.detector.decode_model_output(detection_output, self.confidence_threshold)
        total_time = timer() - start

        if detection_output is None:
            return None
        detection_label, detection_conf, detection_boxes = detection_output
        if debug:
            print(detection_label, detection_conf)
        
        return self.__sort_labels(detection_label, detection_conf, detection_boxes) + (total_time,)

    def run_with_defense(self, defense_type, attack_type = "None", generate_attack = True):
        defense_model = self.defenses[defense_type]
        defense_model.eval()

        if attack_type == "None":
            images = torch.from_numpy(self.preprocessed_image[None, :, :, :]).to(self.device)
            start = timer()
            with torch.no_grad():
                denoised_images = images - defense_model(images)
                
        elif generate_attack:
            images = torch.from_numpy(self.preprocessed_image[None, :, :, :]).to(self.device)
            attack: AttackBase = self.attacks[attack_type]
            attack.model = self.detector.model
            perturbed_images = attack.generate_attack(images)
            start = timer()
            with torch.no_grad():
                denoised_images = perturbed_images - defense_model(perturbed_images)
        
        elif not generate_attack:
            start = timer()
            with torch.no_grad():
                denoised_images = self.detector_attacked_images - defense_model(self.detector_attacked_images)

        detection_output = self.detector.get_model_output(denoised_images)[0]
        detection_output = self.detector.decode_model_output(detection_output, self.confidence_threshold)
        total_time = timer() - start
        
        if detection_output is None:
            return None
        else:
            # classification_labels, classification_conf, detection_boxes =  detection_output
            return detection_output + (total_time,)

if __name__ == "__main__":
    import os
    import cv2
    test_imgs_path = os.path.join(os.path.dirname(__file__), "../test_imgs")
    test_imgs_path = os.path.relpath(test_imgs_path, os.getcwd())
    img = cv2.imread(os.path.join(test_imgs_path, "test6.png"))
    demo = Demo()
    demo.preprocess(img)
    output = demo.run_without_attack(debug=True)
    print("without attack: ")
    print(output[0], output[1], output[2])
    # output = demo.run_with_defense("HGD","FGSM")
    # print(output[0], output[1], output[2])
    print("with attack: ")
    print(output[0], output[1], output[2])

    # output = demo.run_with_attack("FGSM")
    # print("with fgsm attack: ")
    # print(output[0], output[1], output[2])

    # output = demo.run_with_attack("IT-FGSM")
    # print("with it-fgsm attack: ")
    # print(f"labels: \n{output[0]}")
    # print(f"conf: \n{output[1]}")
    # print(f"boxes: \n{output[2]}")