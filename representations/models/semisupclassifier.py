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
        feature_dim: int,
        writer: SummaryWriter,
        hidden_sizes: list[int] = [256, 128],
    ):
        super(SemiSupervisedClassifier, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.writer = writer

        self.backbone = backbone

        layer_sizes = [feature_dim] + hidden_sizes + [num_classes]

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.classifier = nn.Sequential(*layers)

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
