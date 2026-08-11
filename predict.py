import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from config import *
from models.resnet50 import create_model

# ==========================================
# Load Model
# ==========================================

model = create_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

# ==========================================
# Image Transform
# ==========================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])

# ==========================================
# Prediction Function
# ==========================================

def predict_image(image_path):

    if not os.path.exists(image_path):

        print("Image Not Found")

        return

    image = Image.open(image_path).convert("RGB")

    img = transform(image)

    img = img.unsqueeze(0)

    img = img.to(DEVICE)

    with torch.no_grad():

        output = model(img)

        probability = torch.softmax(output,1)

        confidence, prediction = torch.max(
            probability,
            1
        )

    predicted_class = CLASS_NAMES[
        prediction.item()
    ]

    print("="*50)

    print("Prediction")

    print("="*50)

    print("Class :",predicted_class)

    print(
        f"Confidence : {confidence.item()*100:.2f}%"
    )

    print("="*50)

    top_prob, top_class = torch.topk(
        probability,
        2
    )

    print("\nTop Predictions\n")

    for i in range(2):

        cls = CLASS_NAMES[
            top_class[0][i]
        ]

        score = top_prob[0][i]*100

        print(
            f"{cls} : {score:.2f}%"
        )

    plt.figure(figsize=(5,5))

    plt.imshow(image)

    plt.title(
        predicted_class
    )

    plt.axis("off")

    plt.show()


# ==========================================
# Main
# ==========================================

if __name__=="__main__":

    path=input(
        "Enter MRI Image Path : "
    )

    predict_image(path)