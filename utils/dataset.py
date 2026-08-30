from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from config import *

# ==========================================
# Transform
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# Load Full Dataset
# ==========================================

full_dataset = datasets.ImageFolder(
    root=DATASET_DIR,
    transform=transform
)

print("Classes :", full_dataset.classes)
print("Total Images :", len(full_dataset))

# ==========================================
# Split Dataset
# ==========================================

train_size = int(0.7 * len(full_dataset))
valid_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, valid_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# ==========================================
# DataLoaders
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Training Images : {len(train_dataset)}")
print(f"Validation Images : {len(valid_dataset)}")
print(f"Testing Images : {len(test_dataset)}")
