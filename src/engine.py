import torch
import numpy as np
from sklearn.metrics import roc_auc_score, fbeta_score


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (signals, targets) in enumerate(dataloader):
        signals, targets = signals.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, targets)

        loss.backward()
        # Gradient clipping as mandated for later phases, good practice now
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for signals, targets in dataloader:
            signals, targets = signals.to(device), targets.to(device)
            outputs = model(signals)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    try:
        macro_auc = roc_auc_score(all_targets, all_preds, average="macro")
        if np.isnan(macro_auc):
            macro_auc = 0.0
    except ValueError:
        macro_auc = 0.0

    binary_preds = (all_preds > 0.5).astype(int)

    # F2 Score prioritizes recall for high-risk pathologies
    f2 = fbeta_score(
        all_targets, binary_preds, beta=2.0, average="macro", zero_division=0
    )

    return total_loss / len(dataloader), macro_auc, f2
