import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from .kmeans import KMeansClusterAssignment

from typing import Tuple, Optional, Dict


logger = logging.getLogger(__name__)


class SemiSupervisedLoss(nn.Module):
    """
    Combined loss for semi-supervised learning with clustering and consistency regularization
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        writer: SummaryWriter,
        lambda_cluster: float = 1.0,
        cluster_rampup_epochs: int = 0,
        cluster_update_freq: int = 1,
        cluster_all: bool = True,
        lambda_consistency: float = 1.0,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.writer = writer

        self.lambda_cluster = lambda_cluster
        self.cluster_rampup_epochs = cluster_rampup_epochs
        self.cluster_update_freq = cluster_update_freq
        self.cluster_all = cluster_all

        self.lambda_consistency = lambda_consistency
        self.temperature = temperature

        self.supervised_loss = nn.CrossEntropyLoss()
        self.cluster_assigner = KMeansClusterAssignment(num_classes, feature_dim)

    def get_cluster_scale(self, epoch: int) -> float:
        """Calculate cluster loss scaling factor using cosine rampup"""
        if epoch >= self.cluster_rampup_epochs:
            return 1.0
        return float(1 - math.cos(math.pi * epoch / self.cluster_rampup_epochs)) / 2.0

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

        # Update cluster assignments periodically
        if batch_idx % self.cluster_update_freq == 0:
            if self.cluster_all:
                # Combine labeled and unlabeled features for better clustering
                all_features = torch.cat(
                    [features_labeled.detach(), features_unlabeled.detach()], dim=0
                )
                cluster_assignments = self.cluster_assigner.update_clusters(all_features)
                # Only use assignments for unlabeled data
                cluster_assignments = cluster_assignments[features_labeled.size(0) :]
            else:
                cluster_assignments = self.cluster_assigner.update_clusters(
                    features_unlabeled.detach()
                )
        else:
            cluster_assignments = None

        sup_loss = self.supervised_loss(logits_labeled, labels)

        if torch.isnan(sup_loss):
            logger.warning(
                f"NaN sup_loss detected. logits_labeled stats: min={logits_labeled.min():.3f}, max={logits_labeled.max():.3f}"
            )

        loss_dict = {"supervised": sup_loss.item()}
        total_loss = sup_loss

        # Clustering loss on unlabeled data
        if logits_unlabeled is not None and cluster_assignments is not None:
            # Handle potential batch size mismatch
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

            total_loss += self.lambda_cluster * cluster_scale * cluster_loss
            loss_dict["clustering"] = cluster_loss.item()

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
