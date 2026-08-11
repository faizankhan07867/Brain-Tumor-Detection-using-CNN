import torch
import numpy as np


class EarlyStopping:

    def __init__(
        self,
        patience=7,
        verbose=True,
        delta=0,
        path="outputs/best_model.pth"
    ):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:

            self.counter += 1

            print(
                f"EarlyStopping Counter : {self.counter} / {self.patience}"
            )

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        if self.verbose:

            print(
                f"Validation Loss Improved "
                f"({self.val_loss_min:.6f} → {val_loss:.6f})"
            )

            print("Saving Best Model...\n")

        torch.save(
            model.state_dict(),
            self.path
        )

        self.val_loss_min = val_loss