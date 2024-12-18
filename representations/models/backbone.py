import torch
import torch.nn as nn
from typing import Optional, Tuple

class BackboneFeatureExtractor(nn.Module):
    def __init__(self, base_model: nn.Module):
        """
        Initialize a backbone feature extractor.
        
        Args:
            base_model (nn.Module): Pre-trained backbone model
        """
        super().__init__()
        self.backbone = base_model
        self.feature_dim = self._get_feature_dimension()

    def _get_feature_dimension(self) -> int:
        """
        Automatically determine the feature dimension of the backbone.
        
        Returns:
            int: Dimension of the penultimate layer features
        """
        # Placeholder - will need to be customized based on specific backbone
        return 512

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from the input.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            Tuple containing:
            - Penultimate layer features
            - Final layer output (logits)
        """
        # Iterate through backbone layers to extract features
        features = x
        for name, module in self.backbone.named_children():
            if name != 'fc' and name != 'classifier':
                features = module(features)
        
        # Extract penultimate layer features
        penultimate_features = features
        
        # Get final classification output
        logits = self.backbone.fc(features) if hasattr(self.backbone, 'fc') else \
                 self.backbone.classifier(features)
        
        return penultimate_features, logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature embedding for clustering or downstream tasks.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Feature embedding
        """
        features, _ = self.forward(x)
        return features
