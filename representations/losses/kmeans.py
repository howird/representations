import logging

import torch
import torch.nn.functional as F
from torch import Tensor


logger = logging.getLogger(__name__)


class KMeansClusterAssignment:
    """Compute and update cluster assignments for unlabeled data"""

    def __init__(self, num_classes: int, feature_dim: int, max_iterations: int = 10):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.centroids = None
        self.max_iterations = max_iterations

    @torch.no_grad()
    def update_clusters(self, features: Tensor) -> Tensor:
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

        # Normalize features
        features_norm = F.normalize(features, dim=1)

        # Initialize centroids if not exists
        if self.centroids is None:
            indices = torch.randperm(features_norm.size(0))[: self.num_classes]
            self.centroids = features_norm[indices].clone()

        for _ in range(self.max_iterations):
            # Compute distances to centroids
            dists = torch.cdist(features_norm, self.centroids)

            # Assign to nearest centroid
            assignments = dists.argmin(dim=1)

            # Update centroids
            new_centroids = torch.zeros_like(self.centroids)
            for k in range(self.num_classes):
                mask = assignments == k
                if mask.any():
                    cluster_features = features_norm[mask]
                    if not torch.isnan(cluster_features).any():
                        new_centroids[k] = F.normalize(cluster_features.mean(0), dim=0)
                else:
                    new_centroids[k] = self.centroids[k]

            if torch.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return assignments
