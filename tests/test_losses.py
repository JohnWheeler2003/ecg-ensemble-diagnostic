import pytest
import torch
import torch.nn as nn
from src.losses import AsymmetricLoss
from src.models.inception import ECGInceptionTime
import src.config as config

# Constants
CHANNELS = config.CHANNELS
SEQ_LEN = config.SEQ_LEN
NUM_CLASSES = len(config.TARGET_CLASSES)

# Test specific constants
TEST_BATCH_SIZE = 16  # smaller for fast, low-memory unit testing


@pytest.fixture
def loss_fn():
    return AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)


def test_extreme_logits(loss_fn):
    """Verify that extreme logits do not cause NaN or Inf via log(0) explosions"""
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    # Extreme positive and negative logits
    extreme_logits = torch.tensor([[1e4, -1e4], [-1e4, 1e4]])

    loss = loss_fn(extreme_logits, targets)

    assert not torch.isnan(loss), "Loss evaluated to NaN on extreme logits."
    assert not torch.isinf(loss), "Loss evaluated to Inf on extreme logits."
    # With perfect extreme predictions, loss should approach 0
    assert loss.item() < 1e-4


def test_independent_class_penalizations(loss_fn):
    """Verify that multi-label classes are penalized independently"""
    targets = torch.tensor([[1.0, 1.0, 0.0]])

    # Class 0 is perfect, Class 1 is terrible, Class 2 is perfect true negative
    logits = torch.tensor([[10.0, -10.0, -10.0]], requires_grad=True)

    loss = loss_fn(logits, targets)
    loss.backward()

    assert not torch.isnan(loss)
    assert (
        loss.item() > 0.0
    )  # Loss should be > 0 due to the terrible class 1 prediction


def test_full_pipeline_gradient_flow(loss_fn):
    """Pre-Loop Integration Smoke test
    Strict verification that gradients flow from ASL through entire ECGInceptionTime graph"""
    # Instantiate real production model and loss function
    model = ECGInceptionTime(in_channels=CHANNELS, num_classes=NUM_CLASSES)

    # Generate dummy ECG signal data and multi-hot targets with input shape (Batch, Channels, Sequence Length)
    dummy_ecg = torch.randn(TEST_BATCH_SIZE, CHANNELS, SEQ_LEN, requires_grad=True)
    dummy_targets = torch.randint(0, 2, (TEST_BATCH_SIZE, NUM_CLASSES)).float()

    # Forward pass through InceptionTime architecture
    logits = model(dummy_ecg)
    loss = loss_fn(logits, dummy_targets)

    # Backpropagation
    loss.backward()

    # Verify entire graph is connected

    # Check final classification head (ensures loss connects to the model)
    final_layer = next(
        m for m in reversed(list(model.modules())) if isinstance(m, nn.Linear)
    )
    assert final_layer.weight.grad is not None, (
        "Integration Failure: Gradients did not reach the classification head."
    )
    assert not torch.all(final_layer.weight.grad == 0), "Gradients at the head are 0."

    # Check very first convolutional layer (ensures gradients navigated the Inception blocks)
    first_conv_layer = next(m for m in model.modules() if isinstance(m, nn.Conv1d))
    assert first_conv_layer.weight.grad is not None, (
        "Integration failure. Gradients failed to reach the initial convolution layers."
    )

    # Check raw input tensor (end to end differentiability)
    assert dummy_ecg.grad is not None, (
        "Integration Failure: Gradients failed to reach the input tensor."
    )
    assert not torch.all(dummy_ecg.grad == 0), "Input tensor gradients are zero."
