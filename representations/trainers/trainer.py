from ..models.semisupclassifier import SemiSupervisedClassifier
from ..models.backbones import ResNet50Backbone, EfficientNetBackbone, BasicCNNBackbone
from ..losses.semisuploss import SemiSupervisedLoss
from ..datasets.imagenette import ImagenetteDataModule

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from typing import Tuple, Dict, List


class SemiSupervisedTrainer:
    def __init__(
        self,
        num_classes: int,
        labeled_ratio: float = 0.1,
        learning_rate: float = 0.001,
        loss_args: dict = {},
        model_args: dict = {},
        validation_freq: int = 10,
        log_dir: str = "runs",
        exp_name: str = "",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labeled_ratio = labeled_ratio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{timestamp}_ratio_{labeled_ratio}" + (
            f"_{exp_name.replace(" ", "_")}" if exp_name else ""
        )
        self.run_dir = os.path.join(log_dir, self.run_name)
        self.writer = SummaryWriter(self.run_dir)

        # Initialize model and loss
        # backbone = ResNet50Backbone(pretrained=True)
        backbone = EfficientNetBackbone(pretrained=False)
        self.model = SemiSupervisedClassifier(
            num_classes, backbone, backbone.feature_dim, self.writer, **model_args
        ).to(self.device)
        self.loss = SemiSupervisedLoss(num_classes, backbone.feature_dim, self.writer, **loss_args)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.labeled_ratio = labeled_ratio
        self.validation_freq = validation_freq

    def split_labeled_unlabeled(self, dataset: torch.utils.data.Dataset) -> Tuple[Subset, Subset]:
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
        loss_components = {
            "supervised": 0.0,
            "clustering": 0.0,
            "consistency": 0.0,
            # "cluster_consistency": 0.0,
        }

        unlabeled_iter = iter(unlabeled_loader)

        for batch_idx, (labeled_imgs, labels) in enumerate(labeled_loader):
            try:
                unlabeled_imgs, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_imgs, _ = next(unlabeled_iter)

            labeled_imgs = labeled_imgs.to(self.device)
            labels = labels.to(self.device)
            unlabeled_imgs = unlabeled_imgs.to(self.device)

            labeled_imgs = data_module.strong_transforms(labeled_imgs).to(self.device)

            unlabeled_aug1 = data_module.weak_transforms(unlabeled_imgs).to(self.device)
            unlabeled_aug2 = data_module.strong_transforms(unlabeled_imgs).to(self.device)

            labeled_features, labeled_logits = self.model(labeled_imgs)
            unlabeled_features, unlabeled_logits = self.model(unlabeled_aug1)
            _, logits_aug2 = self.model(unlabeled_aug2)

            loss, batch_losses = self.loss(
                labeled_features,
                labeled_logits,
                labels,
                epoch,
                batch_idx,
                unlabeled_features,
                unlabeled_logits,
                logits_aug1=unlabeled_logits,
                logits_aug2=logits_aug2,
            )

            # Skip update if loss is nan
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at batch {batch_idx}")
                self.optimizer.zero_grad()
                continue

            # Update model with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track losses and scaling
            total_loss += loss.item()
            for k, v in batch_losses.items():
                loss_components[k] += v

        # Average losses
        num_batches = len(labeled_loader)
        return {
            "total_loss": total_loss / num_batches,
            **{k: v / num_batches for k, v in loss_components.items()},
        }

    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model using mean per-class accuracy"""
        self.model.eval()
        num_classes = self.model.num_classes
        class_correct = torch.zeros(num_classes, device=self.device)
        class_total = torch.zeros(num_classes, device=self.device)

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                _, outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                # Calculate per-class accuracy
                for cls in range(num_classes):
                    cls_mask = labels == cls
                    if cls_mask.sum() > 0:  # Only evaluate if we have samples
                        class_correct[cls] += ((predicted == labels) & cls_mask).sum()
                        class_total[cls] += cls_mask.sum()

        # Calculate mean per-class accuracy
        per_class_acc = torch.where(
            class_total > 0,
            100.0 * class_correct / class_total,
            torch.zeros_like(class_total),
        )
        valid_classes = (class_total > 0).sum()
        mean_accuracy = per_class_acc.sum() / valid_classes

        # Log per-class accuracies
        for cls in range(num_classes):
            if class_total[cls] > 0:
                logger.info(
                    f"Class {cls} accuracy: {per_class_acc[cls]:.2f}% "
                    f"({class_correct[cls]}/{class_total[cls]})"
                )

        return mean_accuracy.item()

    def train(
        self,
        data_module: ImagenetteDataModule,
        num_epochs: int = 100,
        batch_size: int = 64,
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

        # Get validation dataloader
        val_loader = data_module.val_dataloader()

        history = {
            "total_loss": [],
            "supervised": [],
            "clustering": [],
            "consistency": [],
            "val_accuracy": [],
        }

        for epoch in range(num_epochs):
            epoch_losses = self.train_epoch(labeled_loader, unlabeled_loader, data_module, epoch)

            # Record losses
            for k, v in epoch_losses.items():
                history[k].append(v)
                self.writer.add_scalar(f"Loss/{k}", v, epoch)

            # Validate every N epochs
            if (epoch + 1) % self.validation_freq == 0:
                val_accuracy = self.validate(val_loader)
                history["val_accuracy"].append(val_accuracy)
                self.writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
                logger.info(f"Epoch {epoch+1}/{num_epochs}")
                logger.info(f"Total Loss: {epoch_losses['total_loss']:.4f}")
                logger.info(f"Validation Accuracy: {val_accuracy:.2f}%")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs}")
                logger.info(f"Total Loss: {epoch_losses['total_loss']:.4f}")

        # Save model weights in the run directory
        os.makedirs(self.run_dir, exist_ok=True)
        save_path = os.path.join(self.run_dir, "last.pt")
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

        # Close tensorboard writer
        self.writer.close()
        return history
