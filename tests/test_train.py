import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.train import train_model
from src.losses import AsymmetricLoss
from src.config import CHANNELS, SEQ_LEN, TARGET_CLASSES


class DummyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(CHANNELS * SEQ_LEN, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.fc(x)


def test_train_pipeline(tmp_path):
    """Smoke test for training orchestrator.
    tmp_path automatically creates and destroys temporary directory for safe file saving during the test"""

    # Device and Hyperparameters setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_batch_size = 16
    num_classes = len(TARGET_CLASSES)
    epochs = 3
    test_model_name = "test_smoke_model"

    # Generate Fake Data (Noise for X, Random 0s and 1s for Y)
    train_signals = torch.randn(test_batch_size * 2, CHANNELS, SEQ_LEN)
    train_targets = torch.randint(0, 2, (test_batch_size * 2, num_classes)).float()
    train_dataset = TensorDataset(train_signals, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=test_batch_size, shuffle=True)

    val_signals = torch.randn(test_batch_size, CHANNELS, SEQ_LEN)
    val_targets = torch.randint(0, 2, (test_batch_size, num_classes)).float()
    val_dataset = TensorDataset(val_signals, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)

    # Pipeline Components
    model = DummyModel(num_classes=num_classes).to(device)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)

    # Setup Cosine Annealing with Warm Restarts
    batches_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=batches_per_epoch * 1, T_mult=1
    )

    # Run the Orchestrator
    try:
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            model_name=test_model_name,
            num_epochs=epochs,
            patience=2,
            base_save_dir=tmp_path,
        )
    except Exception as e:
        pytest.fail(f"Pipeline crashed during execution: {e}")

    # Assertions
    assert len(history["train_loss"]) == epochs, "History dictionary missed an epoch!"
    assert len(history["val_f2"]) == epochs, "Validation F2 not tracked properly!"
    assert isinstance(trained_model, nn.Module), (
        "Returned object is not a PyTorch model!"
    )

    # Verify EarlyStopping class built the path and saved the file
    expected_file_path = (
        tmp_path / test_model_name / f"{test_model_name}_best_model.pth"
    )
    assert expected_file_path.exists(), (
        f"Model weights were not saved to {expected_file_path}"
    )
