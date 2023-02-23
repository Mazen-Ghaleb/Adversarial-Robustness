import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import os
import torch


class SignClassifierModel(nn.Module):
    def __init__(self) -> None:
        super(SignClassifierModel, self).__init__()
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
        print(x.shape)
        x = self.convs(x)
        x = self.fc(x)
        return x

def get_sign_classifier_model(device):
    # get the relative path of the current script to the working directory
    dir_relative_path = os.path.relpath(
        os.path.dirname(__file__), os.getcwd())
    # get the path of the model and the expirement script
    model_path = os.path.join(dir_relative_path, "sign_classifier.pth")
    model = SignClassifierModel()
    model.load_state_dict(torch.load(model_path))
    return model.to(device)
