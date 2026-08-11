import torch.nn as nn
from utils.early_stopping import EarlyStopping

model = nn.Linear(10,2)

early = EarlyStopping()

losses = [
    0.8,
    0.7,
    0.65,
    0.66,
    0.67,
    0.68,
    0.69,
    0.70
]

for loss in losses:

    early(loss, model)

    if early.early_stop:

        print("Training Stopped")
        break