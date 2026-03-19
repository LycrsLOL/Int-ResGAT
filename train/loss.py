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
        # Cross Entropy Loss (log_softmax + nll_loss)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the probability of the true class (p_t)
        pt = torch.exp(-ce_loss)
        
        # Focal Loss component
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # If alpha is a scalar, apply it to all classes (or just positive/negative in binary)
                # For multi-class, typically alpha is a list/tensor. 
                # If scalar provided here, we might just scale everything (which doesn't help imbalance directly unless binary)
                # However, common implementation treats alpha as weight for the target class.
                # Let's assume alpha is a scalar applied globally or we need class weights.
                # The user requirement says "flexible read focal_loss_alpha".
                # Standard focal loss alpha is for binary. For multi-class, it's often a tensor of weights.
                # If user provides a single float alpha, we apply it. 
                focal_loss = self.alpha * focal_loss
            elif isinstance(self.alpha, (list, torch.Tensor)):
                # If alpha is a list/tensor of size C
                if isinstance(self.alpha, list):
                    self.alpha = torch.tensor(self.alpha, device=inputs.device)
                
                # Gather alpha for the target classes
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
