import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilabelFocalLoss(nn.Module):
    """Focal loss for multi-label EC prediction.

    Inputs are raw logits with shape ``(batch, num_labels)`` and targets are
    multi-hot vectors with the same shape.
    """

    def __init__(self, alpha=None, gamma=2.0, pos_weight=None, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, inputs, targets):
        targets = targets.float()
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=inputs.device, dtype=inputs.dtype)

        bce_loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction="none",
            pos_weight=pos_weight,
        )

        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = ((1.0 - p_t) ** self.gamma) * bce_loss

        if self.alpha is not None:
            alpha = self.alpha
            if isinstance(alpha, (float, int)):
                alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            else:
                alpha = torch.as_tensor(alpha, device=inputs.device, dtype=inputs.dtype).view(1, -1)
                alpha_t = alpha * targets + (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(MultilabelFocalLoss):
    """Backward-compatible alias used by older training scripts."""

