import pytest
import numpy as np
from unittest.mock import patch
from src.augmentations import apply_temporal_jitter, apply_random_lead_masking


@pytest.fixture
def dummy_signal():
    """Provides a standard 1000 x 12 array of ones for easy math verification"""
    return np.ones((1000, 12))


class TestTemporalJitter:
    @patch("src.augmentations.random.randint")
    def test_positive_shift(self, mock_randint, dummy_signal):
        """Test standard positive (right) shift with zero padding"""
        mock_randint.return_value = 10
        jittered = apply_temporal_jitter(dummy_signal, max_shift=50)

        # The first 10 steps should be zero-padded
        assert np.all(jittered[:10, :] == 0.0)
        # The rest should be the original ones
        assert np.all(jittered[10:, :] == 1.0)
        # Shape must remain identical
        assert jittered.shape == (1000, 12)

    @patch("src.augmentations.random.randint")
    def test_negative_shift(self, mock_randint, dummy_signal):
        """Test standard negative (left) shift with zero padding"""
        mock_randint.return_value = -20
        jittered = apply_temporal_jitter(dummy_signal, max_shift=50)

        # The last 20 steps should be zero-padded
        assert np.all(jittered[-20:, :] == 0.0)
        # The rest should be the original ones
        assert np.all(jittered[:-20, :] == 1.0)
        # Shape must remain identical
        assert jittered.shape == (1000, 12)

    @patch("src.augmentations.random.randint")
    def test_zero_shift(self, mock_randint, dummy_signal):
        """Edge Cases: A shift of 0 should return the exact unmodified array"""
        mock_randint.return_value = 0
        jittered = apply_temporal_jitter(dummy_signal, max_shift=50)
        assert np.array_equal(jittered, dummy_signal)


class TestRandomLeadMasking:
    @patch("src.augmentations.random.random")
    @patch("src.augmentations.random.choice")
    @patch("src.augmentations.random.sample")
    def test_masking_applied(self, mock_sample, mock_choice, mock_random, dummy_signal):
        """Test that leads are correctly zeroed out when probability is met"""
        # Force random probability check to pass (< 0.5)
        mock_random.return_value = 0.1
        # Force it to choose 2 leads to mask
        mock_choice.return_value = 2
        # Force it to pick lead index 0 and 11 (first and last leads)
        mock_sample.return_value = [0, 11]

        masked = apply_random_lead_masking(dummy_signal, mask_prob=0.5)

        # Lead 0 and 11 are entirely 0.0
        assert np.all(masked[:, 0] == 0.0)
        assert np.all(masked[:, 11] == 0.0)
        # Lead 1 (and others) should still be 1.o
        assert np.all(masked[:, 1] == 1.0)

    @patch("src.augmentations.random.random")
    def test_masking_bypassed(self, mock_random, dummy_signal):
        """Edge Case: If probability threshold is not met, return unmodified"""
        # Force probability check to fail (> 0.5)
        mock_random.return_value = 0.9

        masked = apply_random_lead_masking(dummy_signal, mask_prob=0.5)
        assert np.array_equal(masked, dummy_signal)
