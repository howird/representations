import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


class ProtoNet(nn.Module):
    """
    Prototypical Network for few-shot learning
    """

    def __init__(
        self,
        writer: SummaryWriter,
        feature_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()
        self.writer = writer
        self.feature_dim = feature_dim

        # Use ResNet backbone but modify for prototypical networks
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        # Remove final linear layer and modify pooling
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning embeddings

        Args:
            x: Input images [B, C, H, W]
        Returns:
            embeddings: Normalized feature embeddings [B, D]
        """
        embeddings = self.backbone(x)
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        return embeddings

    def compute_prototypes(
        self, features_labeled: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class prototypes from support set embeddings

        Args:
            features_labeled: Support set embeddings [N, D]
            labels: Support set labels [N]
        Returns:
            prototypes: Class prototypes [K, D] where K is number of classes
        """
        classes = torch.unique(labels)
        prototypes = []

        for c in classes:
            # Get embeddings for class c
            mask = labels == c
            class_embeddings = features_labeled[mask]
            # Compute mean embedding (prototype) for class c
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)

    def compute_logits(
        self, query_embeddings: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification logits using prototypical distances

        Args:
            query_embeddings: Query embeddings [N, D]
            prototypes: Class prototypes [K, D]
        Returns:
            logits: Classification logits [N, K]
        """
        # Compute negative squared Euclidean distances
        dists = torch.cdist(query_embeddings, prototypes, p=2)
        logits = -dists
        return logits
