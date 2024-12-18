import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Tuple


class SemiSupervisedClassifier(nn.Module):
    """
    Semi-supervised classifier with feature extraction backbone
    """

    def __init__(
        self, num_classes: int, feature_dim: int = 2048, pretrained: bool = True
    ):
        super().__init__()
        # Load pretrained ResNet backbone
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        # Remove final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Projection head for classification
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both features and logits

        Args:
            x: Input images [B, C, H, W]
        Returns:
            features: Feature embeddings [B, D]
            logits: Classification logits [B, num_classes]
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        # Get classification logits
        logits = self.classifier(features)

        return features, logits
