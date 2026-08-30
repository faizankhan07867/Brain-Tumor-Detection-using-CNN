import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from config import *


class TrainingPlots:

    def __init__(self):

        os.makedirs(GRAPH_DIR, exist_ok=True)

    # ======================================
    # Accuracy Plot
    # ======================================

    def accuracy_plot(

        self,

        train_acc,

        val_acc

    ):

        plt.figure(figsize=(8,5))

        plt.plot(

            train_acc,

            label="Train Accuracy",

            linewidth=2

        )

        plt.plot(

            val_acc,

            label="Validation Accuracy",

            linewidth=2

        )

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy (%)")

        plt.title("Training Accuracy")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                GRAPH_DIR,

                "accuracy.png"

            ),

            dpi=300

        )

        plt.close()

    # ======================================
    # Loss Plot
    # ======================================

    def loss_plot(

        self,

        train_loss,

        val_loss

    ):

        plt.figure(figsize=(8,5))

        plt.plot(

            train_loss,

            label="Train Loss",

            linewidth=2

        )

        plt.plot(

            val_loss,

            label="Validation Loss",

            linewidth=2

        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title("Training Loss")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                GRAPH_DIR,

                "loss.png"

            ),

            dpi=300

        )

        plt.close()

    # ======================================
    # ROC Curve
    # ======================================

    def roc_curve_plot(

        self,

        y_true,

        y_score

    ):

        y_true = label_binarize(

            y_true,

            classes=list(range(NUM_CLASSES))

        )

        plt.figure(figsize=(7,7))

        for i in range(NUM_CLASSES):

            fpr, tpr, _ = roc_curve(

                y_true[:,i],

                y_score[:,i]

            )

            roc_auc = auc(fpr,tpr)

            plt.plot(

                fpr,

                tpr,

                label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.2f})"

            )

        plt.plot(

            [0,1],

            [0,1],

            linestyle="--"

        )

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                GRAPH_DIR,

                "roc_curve.png"

            ),

            dpi=300

        )

        plt.close()

    # ======================================
    # Precision Recall Curve
    # ======================================

    def precision_recall_plot(

        self,

        y_true,

        y_score

    ):

        y_true = label_binarize(

            y_true,

            classes=list(range(NUM_CLASSES))

        )

        plt.figure(figsize=(7,7))

        for i in range(NUM_CLASSES):

            precision, recall, _ = precision_recall_curve(

                y_true[:,i],

                y_score[:,i]

            )

            plt.plot(

                recall,

                precision,

                label=CLASS_NAMES[i]

            )

        plt.xlabel("Recall")

        plt.ylabel("Precision")

        plt.title("Precision Recall Curve")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                GRAPH_DIR,

                "precision_recall.png"

            ),

            dpi=300

        )

        plt.close()

        print("All Graphs Saved Successfully.")
