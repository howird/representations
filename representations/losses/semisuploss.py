import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from .kmeans import KMeansClusterAssignment, SoftKMeansClusterAssignment

from typing import Tuple, Dict


logger = logging.getLogger(__name__)


class SemiSupervisedLoss(nn.Module):
    """
    Combined loss function for semi-supervised learning with multiple components.

    This loss combines:
    1. Supervised cross-entropy loss on labeled data
    2. Clustering loss to group unlabeled samples (with optional ramp-up)
    3. Consistency loss between augmented views of the same samples
    4. Optional cluster consistency loss between network predictions and cluster assignments

    The relative importance of each component is controlled by lambda parameters.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        writer: SummaryWriter,
        lambda_cluster: float = 1.0,
        cluster_rampup_epochs: int = 0,
        cluster_all: bool = True,
        lambda_aug_consistency: float = 1.0,
        lambda_cluster_consistency: float = 0.0,
        temperature: float = 0.5,
        soft_kmeans: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.writer = writer

        self.lambda_cluster = lambda_cluster
        self.cluster_rampup_epochs = cluster_rampup_epochs
        self.cluster_all = cluster_all

        self.lambda_aug_consistency = lambda_aug_consistency
        self.lambda_cluster_consistency = lambda_cluster_consistency
        self.temperature = temperature

        self.supervised_loss = nn.CrossEntropyLoss()
        self.cluster_assigner = (
            SoftKMeansClusterAssignment(num_classes, feature_dim)
            if soft_kmeans
            else KMeansClusterAssignment(num_classes, feature_dim)
        )

    def get_cluster_scale(self, epoch: int) -> float:
        """
        Calculate the scaling factor for cluster loss using cosine rampup schedule.

        Args:
            epoch: Current training epoch

        Returns:
            scale: Scaling factor between 0 and 1, following a cosine curve from
                  0 to 1 over cluster_rampup_epochs epochs
        """
        if epoch >= self.cluster_rampup_epochs:
            return 1.0
        return float(1 - math.cos(math.pi * epoch / self.cluster_rampup_epochs)) / 2.0

    def consistency_loss(self, logits1, logits2):
        # Convert logits to probabilities with temperature scaling
        probs1 = F.softmax(logits1 / self.temperature, dim=-1)
        probs2 = F.softmax(logits2 / self.temperature, dim=-1)

        # Symmetric KL divergence
        return 0.5 * (
            F.kl_div(probs1.log(), probs2, reduction="batchmean")
            + F.kl_div(probs2.log(), probs1, reduction="batchmean")
        )

    def forward(
        self,
        features_labeled: Tensor,
        logits_labeled: Tensor,
        labels: Tensor,
        epoch: int,
        batch_idx: int,
        features_unlabeled: Tensor,
        logits_unlabeled: Tensor,
        logits_aug1: Tensor,
        logits_aug2: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute the combined semi-supervised learning loss.

        Args:
            features_labeled: Feature embeddings of labeled samples [B, D]
            logits_labeled: Model predictions for labeled samples [B, C]
            labels: Ground truth labels for labeled samples [B]
            epoch: Current training epoch
            batch_idx: Current batch index
            features_unlabeled: Feature embeddings of unlabeled samples [B, D]
            logits_unlabeled: Model predictions for unlabeled samples [B, C]
            logits_aug1: Predictions for first augmented view [B, C]
            logits_aug2: Predictions for second augmented view [B, C]

        Returns:
            total_loss: Combined weighted sum of all loss components
            loss_dict: Dictionary containing individual loss values for monitoring:
                      - 'supervised': Cross-entropy loss on labeled data
                      - 'clustering': K-means clustering loss (if enabled)
                      - 'consistency': Augmentation consistency loss (if enabled)
                      - 'cluster_consistency': Cluster prediction consistency (if enabled)

        Notes:
            - B is batch size, C is number of classes, D is feature dimension
            - Clustering loss uses a ramp-up schedule if cluster_rampup_epochs > 0
            - Consistency losses use temperature scaling for the softmax
            - NaN detection is implemented for all loss components
        """

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

        sup_loss = self.supervised_loss(logits_labeled, labels)

        if torch.isnan(sup_loss):
            logger.warning(
                f"NaN sup_loss detected. logits_labeled stats: min={logits_labeled.min():.3f}, max={logits_labeled.max():.3f}"
            )

        loss_dict = {"supervised": sup_loss.item()}
        total_loss = sup_loss

        # Clustering loss on unlabeled data
        if self.lambda_cluster != 0.0:
            # Handle potential batch size mismatch
            min_batch_size = min(logits_unlabeled.size(0), cluster_assignments.size(0))
            cluster_loss = F.cross_entropy(
                logits_unlabeled[:min_batch_size], cluster_assignments[:min_batch_size]
            )

            if torch.isnan(cluster_loss):
                logger.warning("NaN loss detected for cluster_loss")

            cluster_scale = self.get_cluster_scale(epoch)
            if (not self.cluster_rampup_epochs == 0) and batch_idx == 0:
                self.writer.add_scalar("Schedule/cluster_scale", cluster_scale, epoch)
                logger.debug(f"Scheduled Cluster Scale: {cluster_scale:.4f}")

            total_loss += self.lambda_cluster * cluster_scale * cluster_loss
            loss_dict["clustering"] = cluster_loss.item()

        # Augmentation Consistency loss
        if self.lambda_aug_consistency != 0.0:
            aug_consistency_loss = self.consistency_loss(logits_aug1, logits_aug2)

            if torch.isnan(aug_consistency_loss):
                logger.warning("NaN loss detected for aug_consistency_loss")

            total_loss += self.lambda_aug_consistency * aug_consistency_loss
            loss_dict["consistency"] = aug_consistency_loss.item()

        # Cluster Consistency loss
        if self.lambda_cluster_consistency != 0.0:
            # Get one-hot assignments from closest centroids
            unlabeled_cluster_logits = self.cluster_assigner.compute_logits(features_unlabeled)
            labeled_cluster_logits = self.cluster_assigner.compute_logits(features_labeled)
            
            # Use cross entropy since we now have one-hot targets
            cluster_consistency_loss = 0.5 * (
                F.cross_entropy(logits_unlabeled, unlabeled_cluster_logits.argmax(dim=1)) +
                F.cross_entropy(logits_labeled, labeled_cluster_logits.argmax(dim=1))
            )

            if torch.isnan(cluster_consistency_loss):
                logger.warning("NaN loss detected for cluster_consistency_loss")

            total_loss += self.lambda_cluster_consistency * cluster_consistency_loss
            loss_dict["cluster_consistency"] = cluster_consistency_loss.item()

        return total_loss, loss_dict
