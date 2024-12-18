import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Dict


class SemiSupervisedLoss(nn.Module):
    """
    Combined loss for semi-supervised learning with clustering and consistency regularization
    """

    def __init__(
        self,
        num_clusters: int,
        lambda_cluster: float = 1.0,
        lambda_consistency: float = 1.0,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.lambda_cluster = lambda_cluster
        self.lambda_consistency = lambda_consistency
        self.temperature = temperature
        self.supervised_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        logits_labeled: Tensor,
        labels: Tensor,
        logits_unlabeled: Optional[Tensor] = None,
        cluster_assignments: Optional[Tensor] = None,
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
        # Supervised cross-entropy loss
        sup_loss = self.supervised_criterion(logits_labeled, labels)
        if torch.isnan(sup_loss):
            print(f"Warning: NaN loss detected for sup_loss")

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
                print(f"Warning: NaN loss detected for cluster_loss")
            total_loss += self.lambda_cluster * cluster_loss
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
                print(f"Warning: NaN loss detected for consistency_loss")

            total_loss += self.lambda_consistency * consistency_loss
            loss_dict["consistency"] = consistency_loss.item()

        return total_loss, loss_dict


class ClusterAssignment:
    """Compute and update cluster assignments for unlabeled data"""

    def __init__(self, num_clusters: int, feature_dim: int):
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.centroids = None

    @torch.no_grad()
    def update_clusters(self, features: Tensor) -> Tensor:
        """
        Update cluster centroids using k-means

        Args:
            features: Feature embeddings [N, D]
        Returns:
            assignments: Cluster assignments [N]
        """
        # Initialize centroids if not exists
        if self.centroids is None:
            indices = torch.randperm(features.size(0))[: self.num_clusters]
            self.centroids = features[indices].clone()

        for _ in range(10):  # Simple k-means iterations
            # Compute distances to centroids
            dists = torch.cdist(features, self.centroids)

            # Assign to nearest centroid
            assignments = dists.argmin(dim=1)

            # Update centroids
            new_centroids = torch.zeros_like(self.centroids)
            for k in range(self.num_clusters):
                if (assignments == k).any():
                    new_centroids[k] = features[assignments == k].mean(0)
                else:
                    # If empty cluster, keep old centroid
                    new_centroids[k] = self.centroids[k]

            # Check convergence
            if torch.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return assignments
