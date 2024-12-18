import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim as optim

import os
from datetime import datetime

from typing import Tuple, Dict, List

from .models import SemiSupervisedClassifier
from .loss import SemiSupervisedLoss, ClusterAssignment
from .dataset import ImagenetteDataModule


class SemiSupervisedTrainer:
    def __init__(
        self,
        num_classes: int,
        labeled_ratio: float = 0.1,
        num_clusters: int = 10,
        feature_dim: int = 2048,
        learning_rate: float = 0.0001,  # Reduced learning rate
        lambda_cluster: float = 0.1,  # Reduced cluster loss weight
        lambda_consistency: float = 0.1,  # Reduced consistency loss weight
        cluster_update_freq: int = 100,
        log_dir: str = "runs",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labeled_ratio = labeled_ratio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"ratio_{labeled_ratio}_{timestamp}"
        self.run_dir = os.path.join(log_dir, self.run_name)
        self.writer = SummaryWriter(self.run_dir)

        # Initialize model and loss
        self.model = SemiSupervisedClassifier(
            num_classes=num_classes, feature_dim=feature_dim
        ).to(self.device)

        self.criterion = SemiSupervisedLoss(
            num_clusters=num_clusters,
            lambda_cluster=lambda_cluster,
            lambda_consistency=lambda_consistency,
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.cluster_assigner = ClusterAssignment(
            num_clusters=num_clusters, feature_dim=feature_dim
        )

        self.labeled_ratio = labeled_ratio
        self.cluster_update_freq = cluster_update_freq

    def split_labeled_unlabeled(
        self, dataset: torch.utils.data.Dataset
    ) -> Tuple[Subset, Subset]:
        """Split dataset into labeled and unlabeled subsets"""
        num_samples = len(dataset)
        num_labeled = int(num_samples * self.labeled_ratio)

        indices = torch.randperm(num_samples).tolist()
        labeled_indices = indices[:num_labeled]
        unlabeled_indices = indices[num_labeled:]

        return Subset(dataset, labeled_indices), Subset(dataset, unlabeled_indices)

    def train_epoch(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        data_module: ImagenetteDataModule,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {"supervised": 0.0, "clustering": 0.0, "consistency": 0.0}

        unlabeled_iter = iter(unlabeled_loader)

        for batch_idx, (labeled_imgs, labels) in enumerate(labeled_loader):
            try:
                unlabeled_imgs, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_imgs, _ = next(unlabeled_iter)

            # Move to device
            labeled_imgs = labeled_imgs.to(self.device)
            labels = labels.to(self.device)
            unlabeled_imgs = unlabeled_imgs.to(self.device)

            # Convert tensor back to PIL Image for transforms
            to_pil = transforms.ToPILImage()
            unlabeled_pil = [to_pil(img) for img in unlabeled_imgs]

            # Apply transforms to PIL images
            unlabeled_aug1 = torch.stack(
                [data_module.weak_transforms(img) for img in unlabeled_pil]
            ).to(self.device)
            unlabeled_aug2 = torch.stack(
                [data_module.strong_transforms(img) for img in unlabeled_pil]
            ).to(self.device)

            # Forward passes
            labeled_features, labeled_logits = self.model(labeled_imgs)
            unlabeled_features, unlabeled_logits = self.model(unlabeled_aug1)
            _, logits_aug2 = self.model(unlabeled_aug2)

            # Update cluster assignments periodically
            if batch_idx % self.cluster_update_freq == 0:
                cluster_assignments = self.cluster_assigner.update_clusters(
                    unlabeled_features.detach()
                )

            # Compute loss
            loss, batch_losses = self.criterion(
                logits_labeled=labeled_logits,
                labels=labels,
                logits_unlabeled=unlabeled_logits,
                cluster_assignments=cluster_assignments,
                logits_aug1=unlabeled_logits,
                logits_aug2=logits_aug2,
            )

            # Skip update if loss is nan
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at batch {batch_idx}")
                continue

            # Update model with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            for k, v in batch_losses.items():
                loss_components[k] += v

        # Average losses
        num_batches = len(labeled_loader)
        return {
            "total_loss": total_loss / num_batches,
            **{k: v / num_batches for k, v in loss_components.items()},
        }

    def train(
        self, data_module: ImagenetteDataModule, num_epochs: int = 100, batch_size: int = 64
    ) -> Dict[str, List[float]]:
        """Full training loop"""
        train_dataset = data_module.train_dataloader().dataset
        # Split dataset and create loaders with appropriate transforms
        labeled_dataset, unlabeled_dataset = self.split_labeled_unlabeled(train_dataset)

        # Use strong transforms for labeled data
        labeled_loader = DataLoader(
            labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        # Use original transforms for unlabeled base images
        unlabeled_loader = DataLoader(
            unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        history = {
            "total_loss": [],
            "supervised": [],
            "clustering": [],
            "consistency": [],
        }

        for epoch in range(num_epochs):
            epoch_losses = self.train_epoch(
                labeled_loader, unlabeled_loader, data_module, epoch
            )

            # Record losses
            for k, v in epoch_losses.items():
                history[k].append(v)
                self.writer.add_scalar(f"Loss/{k}", v, epoch)

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Total Loss: {epoch_losses['total_loss']:.4f}")

        # Save model weights in the run directory
        os.makedirs(self.run_dir, exist_ok=True)
        save_path = os.path.join(
            self.run_dir, f"model_ratio_{self.labeled_ratio:.3f}.pt"
        )
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        # Close tensorboard writer
        self.writer.close()
        return history
