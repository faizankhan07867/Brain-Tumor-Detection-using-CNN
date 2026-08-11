from utils.logger import *

log_info("Training Started")

log_epoch(
    epoch=1,
    train_loss=0.4567,
    val_loss=0.3211,
    train_acc=95.12,
    val_acc=96.48
)

log_warning("Learning rate reduced.")

log_error("Example error message.")