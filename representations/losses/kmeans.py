import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


logger = logging.getLogger(__name__)


class SoftKMeansClusterAssignment(nn.Module):
    """
    Compute and update soft cluster assignments using differentiable k-means clustering.
    
    This implementation uses a softmax-based soft assignment approach with a temperature
    parameter to control the softness of the assignments. The cluster centroids are 
    registered as nn.Parameters but with requires_grad=False to ensure updates only
    happen through the explicit k-means update rule.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        max_iterations: int = 10,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create centroids as a parameter but without gradients
        self.centroids = nn.Parameter(torch.zeros(num_classes, feature_dim), requires_grad=False).to(self.device)
        self.initialized = False

    def update_clusters(self, features: Tensor) -> Tensor:
        """
        Update cluster centroids using soft k-means clustering and return soft assignments.

        Args:
            features: Feature embeddings of shape [N, D] where N is the number of samples
                     and D is the feature dimension

        Returns:
            assignments: Soft cluster assignments of shape [N, K] where K is the number
                        of clusters. Each row contains softmax probabilities summing to 1.

        Notes:
            - Centroids are initialized using random samples from the input features
            - Uses temperature-scaled softmax for soft assignments
            - Updates centroids using weighted averages based on soft assignments
            - Normalizes updated centroids to unit length
        """
        # Check for invalid values
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.warning("Invalid values detected in features, skipping cluster update")
            return torch.zeros(features.size(0), dtype=torch.long, device=features.device)

        # Initialize centroids if not done yet
        if not self.initialized:
            indices = torch.randperm(features.size(0))[: self.num_classes]
            self.centroids.data.copy_(features[indices].clone())
            self.initialized = True

        for _ in range(self.max_iterations):
            # Compute distances and convert to similarities
            # Detach centroids to prevent gradients flowing through them
            dists = torch.cdist(features, self.centroids.detach())

            # Compute soft assignments using softmax
            similarities = -dists / self.temperature
            assignments = F.softmax(similarities, dim=1)  # [N, K]

            # Update centroids using soft assignments
            new_centroids = torch.zeros_like(self.centroids)
            for k in range(self.num_classes):
                weights = assignments[:, k].unsqueeze(1)  # [N, 1]
                weighted_sum = (features * weights).sum(0)  # [D]
                weight_sum = weights.sum()
                if weight_sum > 0:
                    new_centroids[k] = F.normalize(weighted_sum / weight_sum, dim=0)
                else:
                    new_centroids[k] = self.centroids[k]

            if torch.allclose(self.centroids, new_centroids):
                break

            self.centroids.data.copy_(new_centroids)

        return assignments

    def compute_logits(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute one-hot classification logits based on closest centroid.

        Args:
            features: Input features of shape [N, D] where N is the number of samples
                     and D is the feature dimension

        Returns:
            logits: One-hot encoded logits of shape [N, K] where K is the number of
                   clusters. Each row contains a 1 at the index of the closest centroid
                   and 0 elsewhere.

        Notes:
            - Uses Euclidean distance to find closest centroid
            - Detaches centroids to prevent gradient flow
            - Returns hard one-hot assignments rather than soft probabilities
        """
        # Compute distances
        dists = torch.cdist(features, self.centroids.detach())
        # Get index of closest centroid
        min_indices = dists.argmin(dim=1)
        # Create one-hot encoding
        logits = torch.zeros_like(dists)
        logits.scatter_(1, min_indices.unsqueeze(1), 1)
        return logits


class KMeansClusterAssignment:
    """Compute and update cluster assignments for unlabeled data"""

    def __init__(self, num_classes: int, feature_dim: int, max_iterations: int = 10):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.centroids = None
        self.max_iterations = max_iterations

    @torch.no_grad()
    def update_clusters(self, features: Tensor, normalize: bool = True) -> Tensor:
        """
        Update cluster centroids using k-means on all available features

        Args:
            features: Combined labeled and unlabeled feature embeddings [N, D]
        Returns:
            assignments: Cluster assignments [N]
        """
        # Check for invalid values
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.warning("Invalid values detected in features, skipping cluster update")
            return torch.zeros(features.size(0), dtype=torch.long, device=features.device)

        if normalize:
            features = F.normalize(features, dim=1)

        # Initialize centroids if not exists
        if self.centroids is None:
            indices = torch.randperm(features.size(0))[: self.num_classes]
            self.centroids = features[indices].clone()

        for _ in range(self.max_iterations):
            # Compute distances to centroids
            dists = torch.cdist(features, self.centroids)

            # Assign to nearest centroid
            assignments = dists.argmin(dim=1)

            # Update centroids
            new_centroids = torch.zeros_like(self.centroids)
            for k in range(self.num_classes):
                mask = assignments == k
                if mask.any():
                    cluster_features = features[mask]
                    if not torch.isnan(cluster_features).any():
                        new_centroids[k] = F.normalize(cluster_features.mean(0), dim=0)
                else:
                    new_centroids[k] = self.centroids[k]

            if torch.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return assignments

    def compute_logits(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute classification logits using KNN distances

        Args:
            features: [N, D]
        Returns:
            logits: Classification logits [N, K]
        """
        # Compute negative squared Euclidean distances
        dists = torch.cdist(features, self.centroids.detach())
        logits = -dists
        return logits
