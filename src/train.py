import torch
import numpy as np
import copy
from src.engine import train_epoch, validate_epoch
from src.config import PATIENCE, NUM_EPOCHS, PROJECT_ROOT


class EarlyStopping:
    """EarlyStopping halts the training if validation F2 score doesn't improve after a given patience
    We track F2 to prioritize recall and maintain "Clinical Safety Net"""

    def __init__(self, patience=PATIENCE, delta=0.001, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_max = -np.inf
        self.best_model_state = None

    def __call__(self, val_f2, model):
        score = val_f2

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_f2, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_f2, model)
            self.counter = 0

    def save_checkpoint(self, val_f2, model):
        """Saves model when validation F2 increases."""
        if self.verbose:
            print(
                f"Validation F2 increased ({self.val_max:.4f} --> {val_f2:.4f}). Saving model ..."
            )
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_max = val_f2


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    model_name,
    num_epochs=NUM_EPOCHS,
    patience=PATIENCE,
    base_save_dir=PROJECT_ROOT / "saved_data",
):
    model_saved_dir = base_save_dir / model_name
    model_saved_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    history = {"train_loss": [], "val_loss": [], "val_f2": [], "val_auc": [], "lrs": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # Train Phase
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        # Validation Phase
        val_loss, val_auc, val_f2 = validate_epoch(model, val_loader, criterion, device)

        # Metrics
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Val Loss:   {val_loss:.4f} | Val F2: {val_f2:.4f} | Val AUC: {val_auc:.4f}"
        )
        print(f"Current LR: {current_lr:.6f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f2"].append(val_f2)
        history["val_auc"].append(val_auc)
        history["lrs"].append(current_lr)

        # Early Stopping Check (Driven by F2)
        early_stopping(val_f2, model)
        if early_stopping.early_stop:
            print(
                f"Early stopping triggered due to plateauing F2 score for {model_name}!"
            )
            break

    # Load best model state and save
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

        file_name = f"{model_name}_best_model.pth"
        best_model_path = model_saved_dir / file_name

        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved to {best_model_path}")

    return model, history
