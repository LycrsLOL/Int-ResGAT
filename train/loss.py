import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification to handle class imbalance.
    
    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        
    Args:
        alpha (float, optional): Weighting factor for the rare class (0 < alpha < 1). 
                                 Can also be a list of weights for each class.
        gamma (float, optional): Focusing parameter (gamma >= 0).
        reduction (str, optional): 'mean', 'sum' or 'none'.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-ce_loss)
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                focal_loss = self.alpha * focal_loss
            elif isinstance(self.alpha, (list, torch.Tensor)):
                if isinstance(self.alpha, list):
                    self.alpha = torch.tensor(self.alpha, device=inputs.device)

                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
