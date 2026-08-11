import logging
import os
from datetime import datetime

from config import LOG_DIR

# ==========================
# Create Log Directory
# ==========================

os.makedirs(LOG_DIR, exist_ok=True)

# ==========================
# Log File Name
# ==========================

log_file = os.path.join(
    LOG_DIR,
    f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# ==========================
# Logger Configuration
# ==========================

logger = logging.getLogger("BrainTumorLogger")

logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# File Handler
file_handler = logging.FileHandler(log_file)

file_handler.setFormatter(formatter)

# Console Handler
console_handler = logging.StreamHandler()

console_handler.setFormatter(formatter)

# Avoid duplicate handlers
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# ==========================
# Helper Functions
# ==========================

def log_info(message):
    logger.info(message)

def log_warning(message):
    logger.warning(message)

def log_error(message):
    logger.error(message)

def log_epoch(epoch, train_loss, val_loss, train_acc, val_acc):
    logger.info(
        f"Epoch [{epoch}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}%"
    )