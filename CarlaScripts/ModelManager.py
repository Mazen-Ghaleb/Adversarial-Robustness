# Copyright (c) 2023 Mazen Mostafa Ghaleb, Mostafa Lotfy Mostafa, Safia Medhat Abdulaziz, Youssef Maher Nader
#
# This work is licensed under the terms of the MIT license.
# For a copy, see https://opensource.org/licenses/MIT.

# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================
import matplotlib.image
from timeit import default_timer as timer   

try:
    import cv2
except ImportError:
    raise RuntimeError(
        'cannot import cv2, make sure cv2 package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# Class Imports
from HUD import HUD
from AgentManager import AgentManager

# ==============================================================================
# -- ModelManager --------------------------------------------------------------
# ==============================================================================

class ModelManager(object):
    """Class representing a manager for the classification model, attack methods, and defense methods."""

    def __init__(self):
        self.model_currentTick = 0
        self.model_tickRate = 20
        self.window_first_stats = [True, True, True]

        self.model_flag = False
        self.attack_model_flag = False
        self.defense_model_flag = False
        self.modelPicture_flag = False
        self.modelClassificationPicture_flag = False

        self.detector = None
        self.isOverrideSpeed = False

        self.model_result = None
        self.model_speed = None
        self.model_confidence = None
        self.model_image = self.getEmptyImage()
        self.model_image_window = False

        self.attack_methods = []
        self.attack_currentMethodIndex = 0
        
        self.defense_methods = []
        self.defense_currentMethodIndex = 0

        self.attack_model_result = None
        self.attack_model_speed = None
        self.attack_model_confidence = None
        self.attack_model_image = self.getEmptyImage()
        self.attack_model_image_window = False
        
        self.defense_model_result = None
        self.defense_model_speed = None
        self.defense_model_confidence = None
        self.defense_model_image = self.getEmptyImage()
        self.defense_model_image_window = False
        
    def toggle_modelWindow(self):
        self.model_image_window = not self.model_image_window
        self.window_first_stats[0] = True
    
    def toggle_attackWindow(self):
        self.attack_model_image_window = not self.attack_model_image_window
        self.window_first_stats[1] = True
        
    def toggle_defenseWindow(self):
        self.defense_model_image_window = not self.defense_model_image_window
        self.window_first_stats[2] = True
        
    def set_model_flag(self, flag, vehicle_camera, hud: HUD, agentManager:AgentManager):  # True to activate model and False to stop model
        if (flag):
            vehicle_camera.listen(
                lambda image: self.calculate_model(image, hud, agentManager))
        else:
            vehicle_camera.stop()
            self.model_result = None
        self.model_flag = flag

    def set_attack_model_flag(self, flag:bool):  # True to activate attack model and False to stop attack model
        self.attack_model_flag = flag
    
    def set_defense_model_flag(self, flag:bool):  # True to activate defense model and False to stop defense model
        self.defense_model_flag = flag
            
    def to_bgr_array(self, image):
        """Convert a CARLA raw image to a BGR numpy array."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        return array

    def drawBoundingBox(self, boxes, image: np.ndarray, classifications, confidences, color =(0, 255, 0), font = cv2.FONT_HERSHEY_SIMPLEX, thickness = 2):
        for box,classification,confidence in zip(boxes,classifications,confidences):
            detected_image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)
            detected_image = cv2.putText(detected_image, "{}: {:.2f} %".format(int(classification), confidence*100), 
            (int(box[0]), int(box[1] - 5)), font, 0.6, color, 1)
        return detected_image

    def calculate_model(self, image, hud: HUD, agentManager):
        total_start = timer()
        if (self.model_currentTick == 0):  # Performs the calculation every self.model_tickRate ticks
            image = self.to_bgr_array(image)
            self.detector.preprocess(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.modelPicture_flag:
                matplotlib.image.imsave('../out/test.png', image)
                hud.notification("Saved current view of Speed-limit Sign detection")
                self.modelPicture_flag = False

            self.calculate_classification(hud, np.array(image, copy=True))
            
            if self.defense_model_flag and self.attack_model_flag:
                self.calculate_attack(np.array(image, copy=True))
                self.calculate_defense(np.array(image, copy=True))
                self.calculate_overrideSpeed(agentManager, self.defense_model_speed)
            elif self.defense_model_flag:
                self.calculate_defense(np.array(image, copy=True))
                self.calculate_overrideSpeed(agentManager, self.defense_model_speed)
            elif self.attack_model_flag:
                self.calculate_attack(np.array(image, copy=True))
                self.calculate_overrideSpeed(agentManager, self.attack_model_speed)
            else:
                self.calculate_overrideSpeed(agentManager, self.model_speed)

            print("{:<25}".format("Total Model time"),": {:.3f}s".format(timer()-total_start))
            
        self.model_currentTick = (self.model_currentTick+1) %self.model_tickRate

    def calculate_classification(self, hud, image):
        detection_start = timer()
        self.model_result = self.detector.run_without_attack(debug=True)

        if self.model_result is not None:
            self.model_speed = self.model_result[0][0]
            self.model_confidence = self.model_result[1][0]
            self.model_image = self.drawBoundingBox(self.model_result[2], image, self.model_result[0], self.model_result[1])

            if self.modelClassificationPicture_flag:
                matplotlib.image.imsave('../out/bounding_test.png', self.model_image)
                hud.notification("Saved classified view of Speed-limit Sign detection")
                self.modelClassificationPicture_flag = False

            print("{:<25}".format("Classification Model time"),": {:.3f}s".format(timer()-detection_start),
            "Label:{:<3} Confidence:{:.3f}".format(int(self.model_speed), self.model_confidence))
        else:
            self.model_speed = None
            self.model_confidence = None
            self.model_image = self.getEmptyImage()
            print("{:<25}".format("Classification Model time"),": {:.3f}s No Sign Detected".format(timer()-detection_start))

    def calculate_attack(self, image):
        attack_start = timer()
        self.attack_model_result = self.detector.run_with_attack(self.attack_methods[self.attack_currentMethodIndex])
        
        if self.attack_model_result is not None:
            self.attack_model_speed = self.attack_model_result[0][0]
            self.attack_model_confidence = self.attack_model_result[1][0]
            self.attack_model_image = self.drawBoundingBox(self.attack_model_result[2], image, self.attack_model_result[0], self.attack_model_result[1])
            
            print("{:<25}".format("{} Attack Model time".format(self.attack_methods[self.attack_currentMethodIndex])),
            ": {:.3f}s".format(timer()-attack_start),
            "Label:{:<3} Confidence:{:.3f}".format(int(self.attack_model_speed), self.attack_model_confidence))
        else:
            self.attack_model_speed = None
            self.attack_model_confidence = None
            self.attack_model_image = self.getEmptyImage()
            print("{:<25}".format("{} Attack Model time".format(self.attack_methods[self.attack_currentMethodIndex])),
            ": {:.3f}s No Sign Detected".format(timer()-attack_start))

    def calculate_defense(self, image):
        defense_start = timer()
        self.defense_model_result = self.detector.run_with_defense(self.defense_methods[self.defense_currentMethodIndex],
                                                                    self.attack_methods[self.attack_currentMethodIndex])
        
        if self.defense_model_result is not None:
            self.defense_model_speed = self.defense_model_result[0][0]
            self.defense_model_confidence = self.defense_model_result[1][0]
            self.defense_model_image = self.drawBoundingBox(self.defense_model_result[2], image, self.defense_model_result[0], self.defense_model_result[1])
            
            print("{:<25}".format("{} Defense Model time".format(self.defense_methods[self.defense_currentMethodIndex])),
            ": {:.3f}s".format(timer()-defense_start),
            "Label:{:<3} Confidence:{:.3f}".format(int(self.defense_model_speed), self.defense_model_confidence))
        else:
            self.defense_model_speed = None
            self.defense_model_confidence = None
            self.defense_model_image = self.getEmptyImage()
            print("{:<25}".format("{} Defense Model time".format(self.defense_methods[self.defense_currentMethodIndex])),
            ": {:.3f}s No Sign Detected".format(timer()-defense_start))

    def calculate_overrideSpeed(self, agentManager:AgentManager, detectedSpeed):
        if self.isOverrideSpeed:
            if detectedSpeed:
                print("{:<25}".format("Overriding the speed with"),": {:.3f} km/h".format(detectedSpeed))
                # Over 3.6 to convert it from km/h to m/s because constant velocity takes it in m/s
                #SpeedOfOverride = detectedSpeed /3.6
                agentManager.agent.set_target_speed(detectedSpeed)
                            
    def getEmptyImage(self):
        return np.zeros((600, 800, 3), dtype = np.uint8)