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
    Combined loss for semi-supervised learning with clustering and consistency regularization
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
        # self.cluster_assigner = (
        #     SoftKMeansClusterAssignment(num_classes, feature_dim)
        #     if soft_kmeans
        #     else KMeansClusterAssignment(num_classes, feature_dim)
        # )

    def get_cluster_scale(self, epoch: int) -> float:
        """Calculate cluster loss scaling factor using cosine rampup"""
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

        # if self.cluster_all:
        #     # Combine labeled and unlabeled features for better clustering
        #     all_features = torch.cat(
        #         [features_labeled.detach(), features_unlabeled.detach()], dim=0
        #     )
        #     cluster_assignments = self.cluster_assigner.update_clusters(all_features)
        #     # Only use assignments for unlabeled data
        #     cluster_assignments = cluster_assignments[features_labeled.size(0) :]
        # else:
        #     cluster_assignments = self.cluster_assigner.update_clusters(
        #         features_unlabeled.detach()
        #     )

        sup_loss = self.supervised_loss(logits_labeled, labels)

        if torch.isnan(sup_loss):
            logger.warning(
                f"NaN sup_loss detected. logits_labeled stats: min={logits_labeled.min():.3f}, max={logits_labeled.max():.3f}"
            )

        loss_dict = {"supervised": sup_loss.item()}
        total_loss = sup_loss

        # # Clustering loss on unlabeled data
        # if self.lambda_cluster != 0.0:
        #     # Handle potential batch size mismatch
        #     min_batch_size = min(logits_unlabeled.size(0), cluster_assignments.size(0))
        #     cluster_loss = F.cross_entropy(
        #         logits_unlabeled[:min_batch_size], cluster_assignments[:min_batch_size]
        #     )

        #     if torch.isnan(cluster_loss):
        #         logger.warning("NaN loss detected for cluster_loss")

        #     cluster_scale = self.get_cluster_scale(epoch)
        #     if (not self.cluster_rampup_epochs == 0) and batch_idx == 0:
        #         self.writer.add_scalar("Schedule/cluster_scale", cluster_scale, epoch)
        #         logger.debug(f"Scheduled Cluster Scale: {cluster_scale:.4f}")

        #     total_loss += self.lambda_cluster * cluster_scale * cluster_loss
        #     loss_dict["clustering"] = cluster_loss.item()

        # # Augmentation Consistency loss
        # if self.lambda_aug_consistency != 0.0:
        #     aug_consistency_loss = self.consistency_loss(logits_aug1, logits_aug2)

        #     if torch.isnan(aug_consistency_loss):
        #         logger.warning("NaN loss detected for aug_consistency_loss")

        #     total_loss += self.lambda_aug_consistency * aug_consistency_loss
        #     loss_dict["consistency"] = aug_consistency_loss.item()

        # # Cluster Consistency loss
        # if self.lambda_cluster_consistency != 0.0:
        #     cluster_consistency_loss = 0.5 * (
        #         self.consistency_loss(
        #             self.cluster_assigner.compute_logits(features_unlabeled), logits_unlabeled
        #         )
        #         + self.consistency_loss(
        #             self.cluster_assigner.compute_logits(features_labeled), logits_labeled
        #         )
        #     )

        #     if torch.isnan(cluster_consistency_loss):
        #         logger.warning("NaN loss detected for cluster_consistency_loss")

        #     total_loss += self.lambda_cluster_consistency * cluster_consistency_loss
        #     loss_dict["cluster_consistency"] = cluster_consistency_loss.item()

        return total_loss, loss_dict
