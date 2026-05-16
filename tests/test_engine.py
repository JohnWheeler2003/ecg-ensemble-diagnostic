import pytest
import torch
from torch.optim.lr_scheduler import StepLR
from src.engine import train_epoch, validate_epoch
from src.models.inception import ECGInceptionTime
import src.config as config
from src.losses import AsymmetricLoss


# Pytest Fixtures
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def num_classes():
    return len(config.TARGET_CLASSES)


@pytest.fixture
def model(device, num_classes):
    return ECGInceptionTime(in_channels=config.CHANNELS, num_classes=num_classes).to(
        device
    )


@pytest.fixture
def dummy_batch(num_classes):
    """Generates a deterministic branch matching config constraints"""
    torch.manual_seed(42)
    signals = torch.randn(config.BATCH_SIZE, config.CHANNELS, config.SEQ_LEN)
    targets = torch.randint(0, 2, (config.BATCH_SIZE, num_classes)).float()
    return signals, targets


@pytest.fixture
def dummy_loader(dummy_batch):
    """Simulates a DataLoader yielding a single batch."""
    return [dummy_batch]


@pytest.fixture
def criterion():
    return AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)


@pytest.fixture
def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)


# Unit Tests
def test_train_epoch_scheduler_step(model, dummy_loader, criterion, optimizer, device):
    """Verifies that train_epoch successfully steps the scheduler per batch"""
    scheduler = StepLR(optimizer, step_size=1)
    initial_lr = optimizer.param_groups[0]["lr"]

    train_epoch(model, dummy_loader, criterion, optimizer, scheduler, device)

    new_lr = optimizer.param_groups[0]["lr"]
    assert new_lr != initial_lr, "train_epoch failed to step the scheduler"


def test_validate_epoch_missing_class_handling(model, dummy_loader, criterion, device):
    """Verifies validate_epoch handles ValueError when batch lacks all classes"""
    signals, _ = dummy_loader[0]
    # Force targets to be all zeros
    zero_targets = torch.zeros(config.BATCH_SIZE, len(config.TARGET_CLASSES)).float()
    edge_case_loader = [(signals, zero_targets)]

    val_loss, macro_auc, f2 = validate_epoch(model, edge_case_loader, criterion, device)

    assert macro_auc == 0.0, f"Expected AUC fallback to 0.0 got {macro_auc}"
    assert isinstance(f2, float), "F2 score calculation failed on missing class targets"


def test_overfit_single_batch(model, dummy_loader, criterion, optimizer, device):
    """Master Test: Verifies the entire mathematical graph converges"""
    initial_loss = None
    final_loss = None

    for epoch in range(100):
        loss = train_epoch(
            model, dummy_loader, criterion, optimizer, scheduler=None, device=device
        )
        if epoch == 0:
            initial_loss = loss
        if epoch == 99:
            final_loss = loss
    assert final_loss < initial_loss, "Loss did not decrease at all."
    assert final_loss < 1e-2, (
        f"Gradient Flow Failure: Final loss ({final_loss}) did not converge near zero."
    )

    model.eval()
    signals, targets = dummy_loader[0]
    with torch.no_grad():
        logits = model(signals.to(device))
        preds = torch.sigmoid(logits)
        binary_preds = (preds > 0.5).float()

        correct_predictions = (binary_preds == targets.to(device)).float()
        accuracy = correct_predictions.mean().item()

    assert accuracy == 1.0, (
        f"Overfit Failure: Model failed to achieve 100% accuracy. Got {accuracy * 100}%"
    )
