import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.
    Aggressively down-weights easy negatives while preserving gradients for rare positive signals"""

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
        logits: unnormalzied predictions fromthe model (Batch, Classes)
        targets: multi-hot ground truth matrix (Batch, Classes)"""
        # Ensure targets are float
        targets = targets.float()
        probabilities = torch.sigmoid(logits)

        # Positive Branch: penalize false negatives
        pt_pos = probabilities
        loss_pos = (
            -targets * ((1 - pt_pos) ** self.gamma_pos) * torch.log(pt_pos + self.eps)
        )

        # Negative Branch : penalize false positives
        pt_neg = probabilities.clone()
        if self.clip > 0:
            pt_neg = (pt_neg - self.clip).clamp(min=0.0)

        loss_neg = (
            -(1 - targets) * (pt_neg**self.gamma_neg) * torch.log(1 - pt_neg + self.eps)
        )

        # Combine and average across the batch and class dimensions
        loss = loss_pos + loss_neg
        return loss.mean()
