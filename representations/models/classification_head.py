import torch
import torchvision.transforms as transforms
import numpy as np
import random

class AugmentationStrategy:
    def __init__(
        self, 
        img_size: int = 224, 
        strength: float = 0.5, 
        p_augment: float = 0.8
    ):
        """
        Configurable augmentation strategy for self-supervised learning
        
        Args:
            img_size (int): Target image size for resizing
            strength (float): Augmentation intensity (0-1)
            p_augment (float): Probability of applying augmentations
        """
        self.img_size = img_size
        self.strength = strength
        self.p_augment = p_augment
        
        self.strong_transform = self._build_strong_augmentation()
        self.weak_transform = self._build_weak_augmentation()
    
    def _build_weak_augmentation(self):
        """
        Create a light augmentation pipeline
        """
        return transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
    
    def _build_strong_augmentation(self):
        """
        Create a comprehensive strong augmentation pipeline
        """
        color_jitter = transforms.ColorJitter(
            brightness=self.strength * 0.4,
            contrast=self.strength * 0.4,
            saturation=self.strength * 0.4,
            hue=self.strength * 0.1
        )
        
        return transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=self.p_augment),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(3 * self.strength), sigma=(0.1, 2.0)),
            transforms.RandomRotation(degrees=int(30 * self.strength)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ])
    
    def strong_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply strong augmentations to input tensor
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            torch.Tensor: Augmented image tensor
        """
        # Convert tensor to PIL Image if needed
        if not isinstance(x, transforms.ToPILImage):
            x = transforms.ToPILImage()(x.squeeze(0) if x.dim() == 4 else x)
        
        return self.strong_transform(x)
    
    def weak_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply weak augmentations to input tensor
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            torch.Tensor: Augmented image tensor
        """
        # Convert tensor to PIL Image if needed
        if not isinstance(x, transforms.ToPILImage):
            x = transforms.ToPILImage()(x.squeeze(0) if x.dim() == 4 else x)
        
        return self.weak_transform(x)
    
    def add_gaussian_noise(self, x: torch.Tensor, noise_factor: float = 0.1) -> torch.Tensor:
        """
        Add Gaussian noise to input tensor
        
        Args:
            x (torch.Tensor): Input image tensor
            noise_factor (float): Noise intensity
        
        Returns:
            torch.Tensor: Noisy image tensor
        """
        noise = torch.randn_like(x) * noise_factor
        return x + noise
import torch
import torchvision.transforms as transforms
import numpy as np
import random

class WeakStrongAugmentations:
    def __init__(
        self, 
        img_size: int = 224, 
        strength: float = 0.5, 
        p_augment: float = 0.8
    ):
        """
        Comprehensive augmentation strategy for weakly supervised learning
        
        Args:
            img_size (int): Target image size
            strength (float): Augmentation intensity (0-1)
            p_augment (float): Probability of applying augmentations
        """
        self.img_size = img_size
        self.strength = strength
        self.p_augment = p_augment
        
        self.weak_transform = self._build_weak_augmentation()
        self.strong_transform = self._build_strong_augmentation()
    
    def _build_weak_augmentation(self):
        """Create light augmentation pipeline"""
        return transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
    
    def _build_strong_augmentation(self):
        """Create comprehensive strong augmentation pipeline"""
        color_jitter = transforms.ColorJitter(
            brightness=self.strength * 0.4,
            contrast=self.strength * 0.4,
            saturation=self.strength * 0.4,
            hue=self.strength * 0.1
        )
        
        return transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=self.p_augment),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(3 * self.strength), sigma=(0.1, 2.0)),
            transforms.RandomRotation(degrees=int(30 * self.strength)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ])
    
    def weak_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply weak augmentations to input tensor"""
        return self._convert_and_transform(x, self.weak_transform)
    
    def strong_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply strong augmentations to input tensor"""
        return self._convert_and_transform(x, self.strong_transform)
    
    def _convert_and_transform(self, x: torch.Tensor, transform) -> torch.Tensor:
        """Helper method to convert tensor and apply transform"""
        if not isinstance(x, transforms.ToPILImage):
            x = transforms.ToPILImage()(x.squeeze(0) if x.dim() == 4 else x)
        return transform(x)
import torch
import torch.nn as nn
import torch.optim as optim
from representations.losses.consistency import ConsistencyLoss
from representations.losses.clustering import ClusteringLoss

class WeaklySupervisedTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        num_classes: int,
        labeled_loader,
        unlabeled_loader,
        config: dict = {}
    ):
        """
        Initialize weakly supervised training pipeline
        
        Args:
            model (nn.Module): Base classification model
            num_classes (int): Number of target classes
            labeled_loader: DataLoader for labeled data
            unlabeled_loader: DataLoader for unlabeled data
            config (dict): Training configuration
        """
        self.model = model
        self.num_classes = num_classes
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        
        # Default configuration
        self.config = {
            'consistency_weight': 1.0,
            'clustering_weight': 0.5,
            'learning_rate': 1e-3,
            'total_epochs': 100
        }
        self.config.update(config)
        
        # Loss modules
        self.consistency_loss = ConsistencyLoss(
            augmentation_strategy=self.model.augmentation_strategy,
            temperature=0.1
        )
        self.clustering_loss = ClusteringLoss(
            num_classes=num_classes, 
            feature_dim=model.backbone.feature_dim
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate']
        )
    
    def train_epoch(self, epoch: int):
        """
        Perform one training epoch with weakly supervised learning
        
        Args:
            epoch (int): Current training epoch
        """
        self.model.train()
        
        # Adaptive temperature scheduling
        current_temperature = self.consistency_loss.adaptive_temperature_schedule(
            current_epoch=epoch, 
            total_epochs=self.config['total_epochs']
        )
        
        # Iterate through labeled and unlabeled data
        for (labeled_batch, labels), unlabeled_batch in zip(self.labeled_loader, self.unlabeled_loader):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Supervised loss on labeled data
            features, predictions = self.model(labeled_batch)
            supervised_loss = nn.CrossEntropyLoss()(predictions, labels)
            
            # Consistency loss on unlabeled data
            consistency_loss = self.consistency_loss.compute_consistency_loss(
                model=self.model, 
                unlabeled_images=unlabeled_batch
            )
            
            # Clustering loss
            clustering_loss = self.clustering_loss.compute_clustering_loss(
                features=features, 
                model_predictions=predictions
            )
            
            # Combine losses
            total_loss = (
                supervised_loss + 
                self.config['consistency_weight'] * consistency_loss +
                self.config['clustering_weight'] * clustering_loss
            )
            
            # Backpropagate and optimize
            total_loss.backward()
            self.optimizer.step()
        
        return total_loss.item()
    
    def train(self):
        """
        Full training procedure
        """
        for epoch in range(self.config['total_epochs']):
            epoch_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}: Loss = {epoch_loss}")
