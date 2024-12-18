import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans

class ClusteringLoss:
    def __init__(self, num_classes: int, feature_dim: int):
        """
        Initialize clustering loss module
        
        Args:
            num_classes (int): Number of target classes
            feature_dim (int): Dimension of feature embeddings
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.kmeans = KMeans(n_clusters=num_classes, n_init=10)

    def compute_cluster_assignments(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute cluster assignments using K-means
        
        Args:
            features (torch.Tensor): Feature embeddings
        
        Returns:
            torch.Tensor: Cluster assignments
        """
        # Convert features to numpy for sklearn
        features_np = features.detach().cpu().numpy()
        
        # Perform K-means clustering
        cluster_labels = self.kmeans.fit_predict(features_np)
        
        return torch.from_numpy(cluster_labels).to(features.device)

    def compute_clustering_loss(
        self, 
        features: torch.Tensor, 
        model_predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute clustering loss using pseudo-labels
        
        Args:
            features (torch.Tensor): Feature embeddings
            model_predictions (torch.Tensor): Model's predicted probabilities
        
        Returns:
            torch.Tensor: Clustering loss
        """
        # Compute cluster assignments
        pseudo_labels = self.compute_cluster_assignments(features)
        
        # Convert to one-hot encoding
        pseudo_labels_onehot = torch.nn.functional.one_hot(
            pseudo_labels, 
            num_classes=self.num_classes
        ).float()
        
        # Compute cross-entropy loss between model predictions and pseudo-labels
        clustering_loss = -(pseudo_labels_onehot * torch.log(model_predictions + 1e-8)).mean()
        
        return clustering_loss

    def soft_cluster_assignments(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignments based on centroid distances
        
        Args:
            features (torch.Tensor): Feature embeddings
        
        Returns:
            torch.Tensor: Soft cluster assignment probabilities
        """
        features_np = features.detach().cpu().numpy()
        centroids = self.kmeans.cluster_centers_
        
        # Compute distances to centroids
        distances = np.sum((features_np[:, np.newaxis] - centroids) ** 2, axis=2)
        soft_assignments = torch.softmax(-torch.from_numpy(distances), dim=1)
        
        return soft_assignments.to(features.device)
