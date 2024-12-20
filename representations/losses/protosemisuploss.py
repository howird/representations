import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from .kmeans import KMeansClusterAssignment

from typing import Tuple, Optional, Dict


logger = logging.getLogger(__name__)


class ProtoSemiSupervisedLoss(nn.Module):
    """
    Combined loss for semi-supervised learning with clustering and consistency regularization
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        writer: SummaryWriter,
        lambda_consistency: float = 1.0,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.writer = writer

        self.lambda_consistency = lambda_consistency
        self.temperature = temperature

        self.cluster_assigner = KMeansClusterAssignment(num_classes, feature_dim)

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
        prototypes = []

        for c in self.num_classes:
            # Get embeddings for class c
            mask = labels == c
            class_embeddings = features_labeled[mask]
            # Compute mean embedding (prototype) for class c
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)

    def forward(
        self,
        features_labeled: Tensor,
        logits_labeled: Tensor,
        labels: Tensor,
        epoch: int,
        batch_idx: int,
        features_unlabeled: Optional[Tensor] = None,
        logits_unlabeled: Optional[Tensor] = None,
        logits_aug1: Optional[Tensor] = None,
        logits_aug2: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute combined semi-supervised loss

        Args:
            logits_labeled: Predictions for labeled samples [B, C]
            labels: Ground truth labels [B]
            logits_unlabeled: Predictions for unlabeled samples [B, C]
            cluster_assignments: Cluster assignments for unlabeled samples [B]
            logits_aug1: Predictions for first augmentation [B, C]
            logits_aug2: Predictions for second augmentation [B, C]

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary containing individual loss components
        """

        prototypes = self.compute_prototypes(features_labeled, labels)
        dists = torch.cdist(features_unlabeled, prototypes)

        # Convert distances to probabilities with temperature scaling
        logits = -dists / self.temperature
        log_probs = F.log_softmax(logits, dim=1)

        min_batch_size = min(logits_unlabeled.size(0), cluster_assignments.size(0))
        cluster_loss = F.cross_entropy(
            logits_unlabeled[:min_batch_size], cluster_assignments[:min_batch_size]
        )

        if torch.isnan(cluster_loss):
            logger.warning("NaN loss detected for cluster_loss")

        cluster_scale = self.get_cluster_scale(epoch)
        if batch_idx == 0:  # Log once per epoch
            self.writer.add_scalar("Schedule/cluster_scale", cluster_scale, epoch)
            logger.debug(f"Scheduled Cluster Scale: {cluster_scale:.4f}")

        loss_dict = {"clustering": cluster_loss.item()}
        total_loss = cluster_loss

        # Consistency loss between augmentations
        if logits_aug1 is not None and logits_aug2 is not None:
            # Convert logits to probabilities with temperature scaling
            probs1 = F.softmax(logits_aug1 / self.temperature, dim=-1)
            probs2 = F.softmax(logits_aug2 / self.temperature, dim=-1)

            # Symmetric KL divergence
            consistency_loss = 0.5 * (
                F.kl_div(probs1.log(), probs2, reduction="batchmean")
                + F.kl_div(probs2.log(), probs1, reduction="batchmean")
            )

            if torch.isnan(consistency_loss):
                logger.warning("NaN loss detected for consistency_loss")

            total_loss += self.lambda_consistency * consistency_loss
            loss_dict["consistency"] = consistency_loss.item()

        return total_loss, loss_dict
