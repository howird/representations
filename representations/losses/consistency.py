import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyLoss:
    def __init__(self, augmentation_strategy, temperature=0.1):
        """
        Initialize consistency loss module
        
        Args:
            augmentation_strategy: Function to generate augmentations
            temperature (float): Temperature scaling for softening predictions
        """
        self.augmentation_strategy = augmentation_strategy
        self.temperature = temperature

    def generate_augmentations(self, x: torch.Tensor) -> tuple:
        """
        Generate multiple augmentations of the same input
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            tuple of augmented tensors
        """
        aug1 = self.augmentation_strategy(x)
        aug2 = self.augmentation_strategy(x)
        return aug1, aug2

    def compute_consistency_loss(
        self, 
        model: nn.Module, 
        unlabeled_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss across augmented views
        
        Args:
            model (nn.Module): Classification model
            unlabeled_images (torch.Tensor): Unlabeled image batch
        
        Returns:
            torch.Tensor: Consistency loss
        """
        # Store original device
        device = unlabeled_images.device
        
        # Batch-wise consistency loss
        consistency_losses = []
        
        for image in unlabeled_images:
            # Generate two augmentations of the same image
            aug1, aug2 = self.generate_augmentations(image)
            
            # Extract features and get predictions
            _, pred1 = model(aug1.unsqueeze(0))
            _, pred2 = model(aug2.unsqueeze(0))
            
            # Apply temperature scaling
            pred1 = F.softmax(pred1 / self.temperature, dim=1)
            pred2 = F.softmax(pred2 / self.temperature, dim=1)
            
            # Compute KL divergence between augmented views
            consistency_loss = F.kl_div(
                torch.log(pred1), 
                pred2, 
                reduction='batchmean'
            )
            
            consistency_losses.append(consistency_loss)
        
        # Average consistency loss across batch
        return torch.stack(consistency_losses).mean()

    def adaptive_temperature_schedule(
        self, 
        current_epoch: int, 
        total_epochs: int
    ) -> float:
        """
        Adaptive temperature scheduling for consistency loss
        
        Args:
            current_epoch (int): Current training epoch
            total_epochs (int): Total training epochs
        
        Returns:
            float: Dynamically adjusted temperature
        """
        # Linear warmup and cooldown of temperature
        if current_epoch < total_epochs // 4:
            # Warm-up phase: start with low temperature
            return max(0.01, self.temperature * (current_epoch / (total_epochs // 4)))
        elif current_epoch > 3 * total_epochs // 4:
            # Cool-down phase: gradually reduce temperature
            return max(0.01, self.temperature * (1 - (current_epoch - 3 * total_epochs // 4) / (total_epochs // 4)))
        else:
            # Stable phase: maintain constant temperature
            return self.temperature
