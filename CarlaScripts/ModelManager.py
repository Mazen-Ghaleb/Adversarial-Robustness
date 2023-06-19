# Copyright (c) 2023 Mazen Mostafa Ghaleb, Mostafa Lotfy Mostafa, Safia Medhat Abdulaziz, Youssef Maher Nader
#
# This work is licensed under the terms of the MIT license.
# For a copy, see https://opensource.org/licenses/MIT.

# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================
import matplotlib.image
import os
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
        self.decouple_flag = False
        self.modelPicture_flag = False
        self.modelClassificationPicture_flag = False
        self.modelRecord_flag = False

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
        
        self.detectedSpeed = 30
        
    def toggle_modelWindow(self):
        "Toggle the model image window"
        self.model_image_window = not self.model_image_window
        self.window_first_stats[0] = True
    
    def toggle_attackWindow(self):
        "Toggle the attack model image window"
        self.attack_model_image_window = not self.attack_model_image_window
        self.window_first_stats[1] = True
        
    def toggle_defenseWindow(self):
        "Toggle the defense model image window"
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
    
    def set_decouple_flag(self, flag:bool):  # True to activate decoupling (Run only one model at a time)
        self.decouple_flag = flag
    
    def set_modelPicture_flag(self, flag:bool):  # True to capture model pictures before classifications
        self.modelPicture_flag = flag
    
    def set_modelRecord_flag(self, flag:bool):  # True to record all future model pictures before classifications
        self.modelRecord_flag = flag
    
    def empty_model_results(self):
        "Empty the model results"
        
        # Empty Classification Results
        self.model_result = None
        self.model_speed = None
        self.model_confidence = None
        self.model_image = self.getEmptyImage()
        
        # Empty Attack Results
        self.attack_model_result = None
        self.attack_model_speed = None
        self.attack_model_confidence = None
        self.attack_model_image = self.getEmptyImage()
        
        # Empty Defense Results
        self.defense_model_result = None
        self.defense_model_speed = None
        self.defense_model_confidence = None
        self.defense_model_image = self.getEmptyImage()

    def to_bgr_array(self, image):
        """Convert a CARLA raw image to a BGR numpy array."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        return array

    def drawBoundingBox(self, boxes, image: np.ndarray, classifications, confidences, color =(255, 165, 0), font = cv2.FONT_HERSHEY_SIMPLEX, thickness = 2):
        """Draws the bounding boxes on the image and returns the image"""
        for box,classification,confidence in zip(boxes,classifications,confidences):
            detected_image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)
            detected_image = cv2.putText(detected_image, "{}: {:.2f} %".format(int(classification), confidence*100), 
            (int(box[0]), int(box[1] - 5)), font, 0.6, color, 1)
        return detected_image

    def calculate_model(self, image, hud: HUD, agentManager):
        """Calculates the model and updates the HUD"""
        total_start = timer()
        if (self.model_currentTick == 0):  # Performs the calculation every self.model_tickRate ticks
            image = self.to_bgr_array(image)
            self.detector.preprocess(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.modelPicture_flag:
                matplotlib.image.imsave(self.next_path('../out/modelCapture/output-%s.png'), image)
                hud.notification("Saved current view of Speed-limit Sign detection")
                self.modelPicture_flag = False
            
            if self.modelRecord_flag:
                matplotlib.image.imsave(self.next_path('../out/carlaDomain/output-%s.png'), image)

            if (self.decouple_flag):

                if self.defense_model_flag and self.attack_model_flag: # Only Defense on Attacked Model
                    self.calculate_defense(np.array(image, copy=True), self.attack_methods[self.attack_currentMethodIndex]) 
                    self.calculate_overrideSpeed(agentManager, self.defense_model_speed)
                elif self.attack_model_flag: # Only Attack on Benign Image
                    self.calculate_attack(np.array(image, copy=True))
                    self.calculate_overrideSpeed(agentManager, self.attack_model_speed)
                elif self.defense_model_flag:
                    self.calculate_defense(np.array(image, copy=True)) # Only Defense on Benign Image
                    self.calculate_overrideSpeed(agentManager, self.defense_model_speed)
                else: # Only Normal Classification on Benign Image
                    self.calculate_classification(hud, np.array(image, copy=True))
                    self.calculate_overrideSpeed(agentManager, self.model_speed)

                print("{:<38}".format("Total Model time"),": {:.3f}s".format(timer()-total_start))
                
            else:
                self.calculate_classification(hud, np.array(image, copy=True))
        
                if self.defense_model_flag and self.attack_model_flag:
                    self.calculate_attack(np.array(image, copy=True))
                    self.calculate_defense(np.array(image, copy=True), self.attack_methods[self.attack_currentMethodIndex], generate_attack=False)
                    self.calculate_overrideSpeed(agentManager, self.defense_model_speed)
                elif self.attack_model_flag:
                    self.calculate_attack(np.array(image, copy=True))
                    self.calculate_overrideSpeed(agentManager, self.attack_model_speed)
                elif self.defense_model_flag:
                    self.calculate_defense(np.array(image, copy=True))
                    self.calculate_overrideSpeed(agentManager, self.defense_model_speed)
                else:
                    self.calculate_overrideSpeed(agentManager, self.model_speed)

                print("{:<38}".format("Total Model time"),": {:.3f}s".format(timer()-total_start))
            
        self.model_currentTick = (self.model_currentTick+1) %self.model_tickRate

    def calculate_classification(self, hud, image):
        """Calculates the classification model and updates the HUD"""
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

            # Print Classification Model time
            print("{:<38}".format("Classification Model time"),": {:.3f}s".format(self.model_result[3]))

            # Print Classification Model Process time
            print("{:<38}".format("Classification Model Process time"),": {:.3f}s".format(timer()-detection_start),
            "Label:{:<3} Confidence:{:.3f}".format(int(self.model_speed), self.model_confidence))
        else:
            self.model_speed = None
            self.model_confidence = None
            self.model_image = self.getEmptyImage()
            print("{:<38}".format("Classification Model Process time"),": {:.3f}s No Sign Detected".format(timer()-detection_start))

    def calculate_attack(self, image):
        """Calculates the attack model and updates the HUD"""
        attack_start = timer()
        self.attack_model_result = self.detector.run_with_attack(self.attack_methods[self.attack_currentMethodIndex])

        if self.attack_model_result is not None:
            self.attack_model_speed = self.attack_model_result[0][0]
            self.attack_model_confidence = self.attack_model_result[1][0]
            self.attack_model_image = self.drawBoundingBox(self.attack_model_result[2], image, self.attack_model_result[0], self.attack_model_result[1])
            
            # Print Attack Model Classification time
            print("{:<38}".format("{} Attack Model Classification time".format(self.attack_methods[self.attack_currentMethodIndex])),
            ": {:.3f}s".format(self.attack_model_result[3]))
            
            # Print Attack Model Process time
            print("{:<38}".format("{} Attack Model Process time".format(self.attack_methods[self.attack_currentMethodIndex])),
            ": {:.3f}s".format(timer()-attack_start),
            "Label:{:<3} Confidence:{:.3f}".format(int(self.attack_model_speed), self.attack_model_confidence))
        else:
            self.attack_model_speed = None
            self.attack_model_confidence = None
            self.attack_model_image = self.getEmptyImage()
            
            # Print Attack Model Process time
            print("{:<38}".format("{} Attack Model Process time".format(self.attack_methods[self.attack_currentMethodIndex])),
            ": {:.3f}s No Sign Detected".format(timer()-attack_start))

    def calculate_defense(self, image, attack_method = "None", generate_attack = True):
        """Calculates the defense model and updates the HUD"""
        defense_start = timer()
        self.defense_model_result = self.detector.run_with_defense(self.defense_methods[self.defense_currentMethodIndex],
                                                                    attack_method, generate_attack)
        
        if self.defense_model_result is not None:
            self.defense_model_speed = self.defense_model_result[0][0]
            self.defense_model_confidence = self.defense_model_result[1][0]
            self.defense_model_image = self.drawBoundingBox(self.defense_model_result[2], image, self.defense_model_result[0], self.defense_model_result[1])
            
            # Print Defense Model Classification time
            print("{:<38}".format("{} Defense Model time".format(self.defense_methods[self.defense_currentMethodIndex])),
            ": {:.3f}s".format(self.defense_model_result[3]))
            
            # Print Defense Model Process time
            print("{:<38}".format("{} Defense Model Process time".format(self.defense_methods[self.defense_currentMethodIndex])),
            ": {:.3f}s".format(timer()-defense_start),
            "Label:{:<3} Confidence:{:.3f}".format(int(self.defense_model_speed), self.defense_model_confidence))
        else:
            self.defense_model_speed = None
            self.defense_model_confidence = None
            self.defense_model_image = self.getEmptyImage()
            
            # Print Defense Model Process time
            print("{:<38}".format("{} Defense Model Process time".format(self.defense_methods[self.defense_currentMethodIndex])),
            ": {:.3f}s No Sign Detected".format(timer()-defense_start))

    def calculate_overrideSpeed(self, agentManager:AgentManager, detectedSpeed):
        """Overrides the speed of the agent if the model has detected a speed limit sign"""
        self.detectedSpeed = detectedSpeed
        if self.isOverrideSpeed:
            if detectedSpeed:
                print("{:<38}".format("Overriding the speed with"),": {:.3f} km/h".format(detectedSpeed))
                agentManager.agent.set_target_speed(detectedSpeed)
                            
    def getEmptyImage(self):
        """Returns an empty image of the same size as the camera image"""
        return np.zeros((600, 800, 3), dtype = np.uint8)
    
    def next_path(self, path_pattern):
        """Finds the next free path in an sequentially named list of files."""
        i = 1

        # First do an exponential search
        while os.path.exists(path_pattern % i):
            i = i * 2

        # Result lies somewhere in the interval (i/2..i]
        # This interval (a..b] and narrow it down until a + 1 = b
        a, b = (i // 2, i)
        while a + 1 < b:
            c = (a + b) // 2 # interval midpoint
            a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

        return path_pattern % b