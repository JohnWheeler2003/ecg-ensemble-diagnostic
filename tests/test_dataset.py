import pytest
import torch
import numpy as np
from unittest.mock import patch
from src.dataset import PTBXLDataset, create_dataloaders


@pytest.fixture
def dummy_data():
    """Generates a tiny dataset of 10 patients"""
    X = np.ones((10, 1000, 12))
    # Create random binary multi-hot labels for 5 classes
    y = np.random.randint(0, 2, size=(10, 5))
    return X, y


class TestPTBXLDataset:
    def test_dataset_length(self, dummy_data):
        X, y = dummy_data
        dataset = PTBXLDataset(X, y)
        assert len(dataset) == 10

    def test_tensor_shape_and_type(self, dummy_data):
        """Verifies transpositions and PyTorch conversions are correct"""
        X, y = dummy_data
        dataset = PTBXLDataset(X, y, is_train=False)

        signal, label = dataset[0]

        # Check PyTorch FloatTensor conversion
        assert isinstance(signal, torch.FloatTensor)
        assert isinstance(label, torch.FloatTensor)

        # Check the Conv1D required transpose (1000, 12) -> (12, 1000)
        assert signal.shape == (12, 1000)
        assert label.shape == (5,)

    @patch("src.dataset.apply_random_lead_masking")
    def test_no_in_place_mutation(self, mock_masking, dummy_data):
        """Ensures that calling __getitem__ does not permanently corrupt the master dataset in RAM"""
        X, y = dummy_data

        # Mock made to deliberately mutate the array passed into it
        def destructive_masking(signal):
            signal[:] = 0.0
            return signal

        mock_masking.side_effect = destructive_masking

        dataset = PTBXLDataset(X, y, is_train=True)

        # Fetch item to trigger destructive masking
        _ = dataset[0]

        # Verify master array X remains entirely composed of 1.0s
        assert np.all(dataset.X == 1.0), (
            "Master dataset was muted! Check the .copy() logic"
        )


class TestDataLoaderIntegration:
    def test_dataloader_smoke_test(self, dummy_data):
        """End-to-End Smoke Test: Simulates feeding train, val, and test data into the dataloaders and pulling the first batch to verify the shapes"""
        X, y = dummy_data

        # Override BATCH_SIZE to 4 locally for thorough testing
        test_batch_size = 4

        train_loader, val_loader, test_loader = create_dataloaders(
            X_train=X,
            y_train=y,
            X_val=X,
            y_val=y,
            X_test=X,
            y_test=y,
            batch_size=test_batch_size,
        )

        # Test Train Loader (Augmentations ON, Shuffle ON)
        # Pull a single batch from the train_loader
        train_signals, train_labels = next(iter(train_loader))

        # Expected shape: (Batch_Size, Channels, Length)
        assert train_signals.shape == (test_batch_size, 12, 1000)
        assert train_labels.shape == (test_batch_size, 5)

        # Verify pin_memory is active
        if torch.cuda.is_available():
            assert train_signals.is_pinned()

        # Test Validation Loader (Augmentations OFF, Shuffle OfF)
        # Pull a single batch from the val_loader
        val_signals, val_labels = next(iter(val_loader))

        # Expected shape: (Batch_Size, Channels, Length)
        assert val_signals.shape == (test_batch_size, 12, 1000)
        assert val_labels.shape == (test_batch_size, 5)

        # Test Test Loader (Augmentations OFF, Shuffle OFF)
        # Pull a single batch from the train_loader
        test_signals, test_labels = next(iter(test_loader))

        # Expected shape: (Batch_Size, Channels, Length)
        assert test_signals.shape == (test_batch_size, 12, 1000)
        assert test_labels.shape == (test_batch_size, 5)
