import pytest
import torch
import numpy as np
import torch.nn as nn
from src.models.inception import InceptionModule, ECGInceptionTime
import src.config as config

# Constants
CHANNELS = config.CHANNELS
SEQ_LEN = config.SEQ_LEN
NUM_CLASSES = len(config.TARGET_CLASSES)

# Test specific Constraints
TEST_BATCH_SIZE = 16  # smaller for fast, low-memory unit testing


@pytest.fixture
def dummy_batch():
    # Shape matching standard PTB-XL parsed outputs: (TEST_BATCH_SIZE, 12, 1000)
    return torch.randn(TEST_BATCH_SIZE, CHANNELS, SEQ_LEN)


class TestInceptionArchitecture:
    """Unit tests for mathematical and dimensional correctness"""

    def test_inception_module_dimensions(self, dummy_batch):
        """Verifies the core block preserves sequence length via 'same' padding"""
        module = InceptionModule(in_channels=CHANNELS, out_channels=32)
        out = module(dummy_batch)

        # Output channels = (32 * 3 conv branches) + 32 (pool branch) = 128
        assert out.shape == (TEST_BATCH_SIZE, 128, SEQ_LEN), (
            "Inception block changed temporal dimensions!"
        )

    def test_full_model_forward_pass(self, dummy_batch):
        """ "Ensures the entire network handles the 12 x 1000 tensor and outputs 5 logits"""
        model = ECGInceptionTime(in_channels=CHANNELS, num_classes=NUM_CLASSES)
        model.eval()

        with torch.no_grad():
            logits = model(dummy_batch)

        assert logits.shape == (TEST_BATCH_SIZE, NUM_CLASSES), (
            f"Expected {(TEST_BATCH_SIZE, NUM_CLASSES)}, got {logits.shape}"
        )
        assert torch.isnan(logits).sum() == 0, "Forward pass procduced NaNs"

    def test_kaiming_initialization_applied(self):
        """Verifies Conv1d weights are initialized with non-zero variance"""
        model = ECGInceptionTime()
        for m in model.modules():
            if isinstance(m, nn.Conv1d):
                # Standard Kaiming Normal creates weights with mean ~0
                # Check they aren't defualt PyTorch uniform initialization (which has different bound) or zeros
                assert torch.std(m.weight).item() > 0.0, (
                    "Weights appear uninitialized (zero variance)."
                )

    def test_prior_informed_bias(self):
        """Verifies the bias mathematics match expected logarithmic outputs"""
        model = ECGInceptionTime(num_classes=NUM_CLASSES)

        # Dummy priors (e.g., NORM is 50%, MI is 20%, etc)
        priors = [0.5, 0.2, 0.1, 0.15, 0.05]
        model.initialize_prior_bias(priors)

        # Calculate expected for index 1 (MI: 0.2)
        # b = ln(0.2/ 0.8) = ln (0.25) = -1.386
        expected_bias_mi = np.log(0.2 / (1.0 - 0.2))

        actual_bias_mi = model.fc.bias.data[1].item()

        assert pytest.approx(actual_bias_mi, 1e-4) == expected_bias_mi, (
            "Prior bias math is incorrect"
        )

    def test_prior_informed_bias_edge_cases(self):
        """Ensures 0% and 100% priors don't cause inf/NaN due to log(0)"""
        model = ECGInceptionTime(num_classes=NUM_CLASSES)
        priors = [0.0, 1.0, 0.5, 0.5, 0.5]

        model.initialize_prior_bias(priors)

        assert torch.isnan(model.fc.bias).sum() == 0, (
            "Edge case priors resulted in NaNs."
        )
        assert torch.isinf(model.fc.bias).sum() == 0, (
            "Edge case priors resulted in Infs."
        )


class TestModelIntegration:
    """Pre-loop Smoke Tests to ensure gradient flow before involving the Training Loop"""

    def test_smoke_backward_pass(self, dummy_batch):
        """Creates a dummy BCE loss, runs a backward pass, and verifies that gradients successfully cascade all the way to the first layer"""
        model = ECGInceptionTime()
        model.train()

        # Dummy labels matching the multi-hot multi-label format
        dummy_labels = torch.randint(0, 2, (TEST_BATCH_SIZE, NUM_CLASSES)).float()

        # Standard BCE Loss as temporary stand in for ASL just to rest graph connections
        criterion = nn.BCEWithLogitsLoss()

        # Forward pass
        logits = model(dummy_batch)
        loss = criterion(logits, dummy_labels)

        # Backward pass
        loss.backward()

        # Check first layer gradients (very first bottleneck conv in the first block)
        first_layer = model.blocks[0].bottleneck

        assert first_layer.weight.grad is not None, (
            "Gradients did not flow to the first layer."
        )
        assert torch.sum(torch.abs(first_layer.weight.grad)) > 0, (
            "First layer gradients are exactly zero (vanishing gradient)."
        )


class TestEnterpriseEdgeCases:
    """Tests for pseudo real-world deployment and CI/CD edge cases"""

    def test_batch_size_one_inference(self):
        """Verifies that BatchNorm layers do not crash during single-patient inference"""
        model = ECGInceptionTime()
        model.eval()

        single_patient_batch = torch.randn(1, CHANNELS, SEQ_LEN)

        with torch.no_grad():
            try:
                logits = model(single_patient_batch)
            except ValueError as e:
                pytest.fail(
                    f"Model crashed on batch size 1. Likely a BatchNorm issue. Error: {e}"
                )
        assert logits.shape == (1, NUM_CLASSES), (
            "Output dimensions are incorrect for single batch"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU to run.")
    def test_device_agnosticism(self, dummy_batch):
        """Verifies the model doesn't have hardcoded CPU/GPU tensors that break on device transfer"""
        model = ECGInceptionTime().to("cuda")
        dummy_batch = dummy_batch.to("cuda")

        logits = model(dummy_batch)

        assert logits.device.type == "cuda", (
            "Model output did not remain on target device."
        )


class TestAdvancedGradientFlow:
    """Deep inspection of backward pass mechanics"""

    def test_shortcut_gradient_activation(self, dummy_batch):
        """Explicitly proves residual projection layers are receiving learning signals"""
        model = ECGInceptionTime(in_channels=CHANNELS)
        model.train()

        # Block 0 requires a projection shortcut because input is 12 channels and output is 128, so we grab that specific 1x1 convolution layer
        projection_conv = model.shortcuts[0][0]

        # Forward and backward pass
        logits = model(dummy_batch)
        loss = logits.sum()  # Dummy scalar loss
        loss.backward()

        assert projection_conv.weight.grad is not None, (
            "Gradients did not reach the projection shortcut."
        )
        assert torch.sum(torch.abs(projection_conv.weight.grad)) > 0, (
            "Projection shortcut gradients are zero."
        )

    def test_frozen_feature_extractor(self, dummy_batch):
        """Verifies standard transfer-oearning mechanics. If blocks are frozen do the gradients still correctly flow to the classification head?"""
        model = ECGInceptionTime()

        # Freeze all Inception blocks
        for param in model.blocks.parameters():
            param.requires_grad = False

        logits = model(dummy_batch)
        loss = logits.sum()
        loss.backward()

        # The classification head should STILL have gradients
        assert model.fc.weight.grad is not None, (
            "Freezing blocks broke the classificaiton head."
        )
        assert model.blocks[0].bottleneck.weight.grad is None, (
            "Frozen layer somehow received gradients."
        )
