import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import *
from models.resnet50 import create_model


class BrainTumorGradCAM:

    def __init__(self):

        self.model = create_model()

        self.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=DEVICE
            )
        )

        self.model.eval()

        # Last convolution layer
        self.target_layers = [
            self.model.model.layer4[-1]
        ]

        self.transform = transforms.Compose([

            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )

        ])

    def generate(self, image_path):

        image = Image.open(image_path).convert("RGB")

        rgb = np.array(image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )) / 255.0

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0).to(DEVICE)

        cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers
        )

        grayscale_cam = cam(
            input_tensor=tensor
        )[0]

        visualization = show_cam_on_image(
            rgb,
            grayscale_cam,
            use_rgb=True
        )

        return visualization

    def save(self, image_path, save_path):

        heatmap = self.generate(image_path)

        cv2.imwrite(
            save_path,
            cv2.cvtColor(
                heatmap,
                cv2.COLOR_RGB2BGR
            )
        )

        print("GradCAM Saved :", save_path)
