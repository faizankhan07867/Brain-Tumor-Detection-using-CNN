import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from config import *


def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    print("="*50)

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print("="*50)

    return accuracy, precision, recall, f1


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=CLASS_NAMES
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        "outputs/confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("Confusion Matrix Saved")


def classification_results(
    y_true,
    y_pred,
    class_names=CLASS_NAMES
):

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    print(report)

    with open(
        "outputs/classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    print("Classification Report Saved")