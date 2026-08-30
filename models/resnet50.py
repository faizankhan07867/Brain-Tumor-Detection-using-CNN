import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import *




import torch
import torch.nn as nn
from torchvision import models
from config import *

class BrainTumorResNet50(nn.Module):

    def __init__(self):

        super().__init__()

        # Load Pretrained ResNet50
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        # Freeze first layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze Layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace FC Layer
        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(

            nn.Linear(in_features,512),

            nn.BatchNorm1d(512),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512,256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256,NUM_CLASSES)

        )

    def forward(self,x):

        return self.model(x)


def create_model():

    model = BrainTumorResNet50()

    model = model.to(DEVICE)

    return model


if __name__ == "__main__":

    model = create_model()

    print(model)

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("="*50)
    print("Total Parameters :", total)
    print("Trainable Parameters :", trainable)
    print("="*50)
