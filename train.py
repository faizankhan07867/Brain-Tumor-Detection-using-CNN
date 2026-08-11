import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import *

from models.resnet50 import create_model

from utils.dataset import (
    train_loader,
    valid_loader
)

from utils.logger import (
    log_info,
    log_epoch
)

from utils.early_stopping import EarlyStopping

from utils.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    classification_results
)

# ==========================================
# TensorBoard
# ==========================================

writer = SummaryWriter(LOG_DIR)

# ==========================================
# Model
# ==========================================

model = create_model()

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.1,

    patience=3,

    verbose=True

)

early_stopping = EarlyStopping(

    patience=PATIENCE,

    path=MODEL_PATH

)

# ==========================================
# History
# ==========================================

train_loss_history = []

val_loss_history = []

train_acc_history = []

val_acc_history = []

best_accuracy = 0

log_info("Training Started")

# ==========================================
# Training Function
# ==========================================

def train_one_epoch(epoch):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    progress = tqdm(

        train_loader,

        desc=f"Epoch {epoch}/{EPOCHS}"

    )

    for images, labels in progress:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        progress.set_postfix(

            loss=loss.item(),

            accuracy=100 * correct / total

        )

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = 100 * correct / total

    train_loss_history.append(epoch_loss)

    train_acc_history.append(epoch_accuracy)

    writer.add_scalar(
        "Train/Loss",
        epoch_loss,
        epoch
    )

    writer.add_scalar(
        "Train/Accuracy",
        epoch_accuracy,
        epoch
    )

    return epoch_loss, epoch_accuracy

# ==========================================
# Validation Function
# ==========================================

def validate(epoch):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    y_true = []

    y_pred = []

    with torch.no_grad():

        progress = tqdm(

            valid_loader,

            desc="Validation"

        )

        for images, labels in progress:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())

            y_pred.extend(predicted.cpu().numpy())

            progress.set_postfix(

                loss=loss.item(),

                accuracy=100 * correct / total

            )

    val_loss = running_loss / len(valid_loader)

    val_accuracy = 100 * correct / total

    val_loss_history.append(val_loss)

    val_acc_history.append(val_accuracy)

    writer.add_scalar(
        "Validation/Loss",
        val_loss,
        epoch
    )

    writer.add_scalar(
        "Validation/Accuracy",
        val_accuracy,
        epoch
    )

    scheduler.step(val_loss)

    early_stopping(val_loss, model)

    global best_accuracy

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(

            model.state_dict(),

            MODEL_PATH

        )

        log_info(
            f"Best Model Saved : {best_accuracy:.2f}%"
        )

    log_epoch(

        epoch,

        train_loss_history[-1],

        val_loss,

        train_acc_history[-1],

        val_accuracy

    )

    calculate_metrics(

        y_true,

        y_pred

    )

    return (

        val_loss,

        val_accuracy,

        y_true,

        y_pred

    )
    
    # ==========================================
# Main Training Loop
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("🧠 Brain Tumor Detection Training Started")
    print("=" * 60)

    start_time = time.time()

    final_true = []
    final_pred = []

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(epoch)

        val_loss, val_acc, y_true, y_pred = validate(epoch)

        final_true = y_true
        final_pred = y_pred

        print(
            f"\nEpoch [{epoch}/{EPOCHS}]"
        )

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy : {train_acc:.2f}%"
        )

        print(
            f"Validation Loss : {val_loss:.4f}"
        )

        print(
            f"Validation Accuracy : {val_acc:.2f}%"
        )

        if early_stopping.early_stop:

            print("\nEarly Stopping Triggered")

            log_info("Training Stopped by Early Stopping")

            break

    # ======================================
    # Save Evaluation
    # ======================================

    plot_confusion_matrix(
        final_true,
        final_pred
    )

    classification_results(
        final_true,
        final_pred
    )

    # ======================================
    # Accuracy Graph
    # ======================================

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,5))

    plt.plot(
        train_acc_history,
        label="Train Accuracy"
    )

    plt.plot(
        val_acc_history,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "accuracy.png"
        )
    )

    plt.close()

    # ======================================
    # Loss Graph
    # ======================================

    plt.figure(figsize=(8,5))

    plt.plot(
        train_loss_history,
        label="Train Loss"
    )

    plt.plot(
        val_loss_history,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "loss.png"
        )
    )

    plt.close()

    writer.close()

    end_time = time.time()

    total_time = (end_time - start_time) / 60

    print("=" * 60)

    print("Training Finished Successfully")

    print(f"Best Validation Accuracy : {best_accuracy:.2f}%")

    print(f"Training Time : {total_time:.2f} Minutes")

    print(f"Model Saved : {MODEL_PATH}")

    print("=" * 60)

    log_info("Training Completed Successfully")