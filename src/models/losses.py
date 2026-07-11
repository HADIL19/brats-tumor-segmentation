"""
Loss functions for multi-class 3D segmentation.

Dice loss directly optimizes overlap (what we're evaluated on), while cross-entropy
gives more stable gradients early in training. Combining them is standard practice
in medical segmentation (nnU-Net and most BraTS-winning solutions use this combo).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss, computed per class and averaged (macro-averaged, so small
    classes like enhancing tumor aren't drowned out by background)."""

    def __init__(self, num_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, D, H, W), target: (B, D, H, W) with integer class labels
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes)
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs + target_onehot, dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss."""

    def __init__(self, num_classes: int, dice_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + (1 - self.dice_weight) * self.ce(logits, target)
