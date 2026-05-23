import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (roc_auc_score, fbeta_score, roc_curve, multilabel_confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve
from src.config import PROJECT_ROOT

def run_inference(model, dataloader, device):
    """Runs a model over a dataloader and returns true labels and probabilities"""
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            
            all_targets.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_targets), np.vstack(all_probs)

def calculate_metrics(y_true, y_prob, threshold=0.5):
    """Calculate Multi-label AUC and F2 scores"""
    y_pred = (y_prob >= threshold).astype(int)
    num_classes = y_true.shape[1]

    # Macro AUC
    try:
        macro_auc = roc_auc_score(y_true, y_prob, average="macro")
    except ValueError:
        macro_auc = float('nan') # Handles edge case if class has only one label type in a batch

    # Macro and Per-class F2 Score
    macro_f2 = fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0)
    per_class_f2 = fbeta_score(y_true, y_pred, beta=2, average=None, zero_division=0)

    # Brier Scores (lower is better)
    per_class_brier = [brier_score_loss(y_true[:, i], y_prob[:, i]) for i in range(num_classes)]
    macro_brier = np.mean(per_class_brier)

    return macro_auc, macro_f2, per_class_f2, macro_brier, per_class_brier