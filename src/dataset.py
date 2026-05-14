import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.config import BATCH_SIZE
from src.augmentations import apply_temporal_jitter, apply_random_lead_masking


class PTBXLDataset(Dataset):
    def __init__(self, X, y, is_train=False):
        self.X = X
        self.y = y
        self.is_train = is_train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        signal = self.X[idx].copy()
        label = self.y[idx]

        if self.is_train:
            # Call augmentation functions
            signal = apply_temporal_jitter(signal)
            signal = apply_random_lead_masking(signal)

        # Transpose for PyTorch Conv1D (It requires (12, 100) (Batch, Channels, Length) instead of (1000, 12))
        signal = np.transpose(signal, (1, 0))

        return torch.FloatTensor(signal), torch.FloatTensor(label)


def create_dataloaders(
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size=BATCH_SIZE
):
    """Wraps the datasets in Pytorch DataLoader objects for optimized batching"""
    train_dataset = PTBXLDataset(X_train, y_train, is_train=True)
    val_dataset = PTBXLDataset(X_val, y_val, is_train=False)
    test_dataset = PTBXLDataset(X_test, y_test, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader, test_loader
