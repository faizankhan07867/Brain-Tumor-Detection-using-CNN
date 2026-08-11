import os
import torch

# ======================================
# Base Path
# ======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================
# Dataset Path
# ======================================

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# ======================================
# Output Paths
# ======================================

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = OUTPUT_DIR
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

DATABASE_PATH = os.path.join(OUTPUT_DIR, "brain_tumor.db")

# ======================================
# Classes
# ======================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

NUM_CLASSES = len(CLASS_NAMES)

# ======================================
# Image Settings
# ======================================

IMAGE_SIZE = 224

# ======================================
# Training Settings
# ======================================

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0      # Windows friendly

# ======================================
# Device
# ======================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ======================================
# Early Stopping
# ======================================

PATIENCE = 7

# ======================================
# Random Seed
# ======================================

SEED = 42

# ======================================
# Create Required Directories
# ======================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("=" * 50)
print("Brain Tumor Detection Configuration")
print("=" * 50)
print("Dataset :", DATASET_DIR)
print("Device :", DEVICE)
print("Epochs :", EPOCHS)
print("Batch Size :", BATCH_SIZE)
print("Learning Rate :", LEARNING_RATE)
print("Classes :", CLASS_NAMES)
print("=" * 50)