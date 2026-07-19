import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedDiceLoss(nn.Module):
    """GeneralizedDiceLoss: weighted Dice loss for imbalanced multi-class segmentation."""

    def __init__(self, smooth: float = 1e-5):
        """Initialize GeneralizedDiceLoss with smoothing factor."""
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        """Compute Generalized Dice loss excluding background class 0."""
        probabilities = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        probabilities = probabilities[:, 1:]
        target_one_hot = target_one_hot[:, 1:]
        dimensions = (0, 2, 3, 4)
        prediction_sum = torch.sum(probabilities, dim=dimensions)
        target_sum = torch.sum(target_one_hot, dim=dimensions)
        intersection_sum = torch.sum(probabilities * target_one_hot, dim=dimensions)
        valid_classes = target_sum > 0
        if not valid_classes.any():
            return probabilities.sum() * 0.0
        prediction_sum = prediction_sum[valid_classes]
        target_sum = target_sum[valid_classes]
        intersection_sum = intersection_sum[valid_classes]
        class_weights = 1.0 / (target_sum.square() + self.smooth)
        numerator = 2.0 * torch.sum(class_weights * intersection_sum)
        denominator = torch.sum(class_weights * (prediction_sum + target_sum))
        generalized_dice = (numerator + self.smooth) / (denominator + self.smooth)
        loss = 1.0 - generalized_dice
        return loss


class DiceCELoss(nn.Module):
    """DiceCELoss: combines GeneralizedDiceLoss and weighted CrossEntropyLoss."""

    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0):
        """Initialize DiceCELoss with configurable Dice and CE loss weights."""
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = GeneralizedDiceLoss()
        class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        """Compute weighted sum of Dice loss and CrossEntropy loss."""
        dice = self.dice_loss(logits, target)
        ce = F.cross_entropy(logits, target.long(), weight=self.class_weights)
        loss = self.dice_weight * dice + self.ce_weight * ce
        return loss
