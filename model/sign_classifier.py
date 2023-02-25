import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import os
import torch
import cv2
import numpy as np

class SignClassifierNet(nn.Module):
    def __init__(self) -> None:
        super(SignClassifierNet, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax(dim=1)

        )

    def forward(self, x: Tensor):
        x = self.convs(x)
        x = self.fc(x)
        return x

def get_sign_classifier_model(device):
    # get the relative path of the current script to the working directory
    dir_relative_path = os.path.relpath(
        os.path.dirname(__file__), os.getcwd())
    # get the path of the model and the expirement script
    model_path = os.path.join(dir_relative_path, "sign_classifier.pth")
    model = SignClassifierNet()
    model.load_state_dict(torch.load(model_path))
    return model.to(device)

class SignClassifier:
    def __init__(self, device) -> None:
        self.model = get_sign_classifier_model(device)

    def preprocess(self,image):
        image = cv2.resize(image, (64, 64))
        image = image.transpose((2, 0, 1))
        return image

    def classify_signs(self, images):
        self.model.eval()
        with torch.no_grad():
            model_outputs = self.model(images).cpu().numpy()
            labels = model_outputs.argmax(1)
            conf = model_outputs[np.arange(model_outputs.shape[0]),labels.ravel()]
            return labels, conf

def classifier_target_generator(outputs):
    labels = outputs.argmax(1)
    targets = F.one_hot(labels, 10).float()
    return targets

def classifier_loss(outputs, targets):
    F.cross_entropy(outputs, targets)