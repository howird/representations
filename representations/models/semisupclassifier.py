import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from typing import Tuple


class SemiSupervisedClassifier(nn.Module):
    """
    Semi-supervised classifier with feature extraction backbone
    """

    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        writer: SummaryWriter,
        feature_dim: int = 2048,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.writer = writer

        self.backbone = backbone
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both features and logits

        Args:
            x: Input images [B, C, H, W]
        Returns:
            features: Feature embeddings [B, D]
            logits: Classification logits [B, num_classes]
        """
        features = self.backbone(x)
        logits = self.classifier(features)

        return features, logits
