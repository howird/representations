from ..models.protonet import ProtoNet
from ..losses.protoloss import ProtoLoss
from ..datasets.imagenette import ImagenetteDataModule

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from typing import Dict, List


class ProtoNetTrainer:
    def __init__(
        self,
        feature_dim: int = 512,
        learning_rate: float = 0.001,
        n_way: int = 5,
        n_support: int = 5,
        n_query: int = 15,
        validation_freq: int = 10,
        log_dir: str = "runs",
        exp_name: str = "",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{timestamp}_{n_way}way_{n_support}shot" + (
            f"_{exp_name.replace(' ', '_')}" if exp_name else ""
        )
        self.run_dir = os.path.join(log_dir, self.run_name)
        self.writer = SummaryWriter(self.run_dir)

        # Episode parameters
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query

        # Initialize model and loss
        self.model = ProtoNet(
            writer=self.writer,
            feature_dim=feature_dim,
        ).to(self.device)

        self.loss_fn = ProtoLoss(
            writer=self.writer,
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.validation_freq = validation_freq

    def train_epoch(
        self,
        train_loader: DataLoader,
        data_module: ImagenetteDataModule,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch using episodic training"""
        self.model.train()
        total_loss = 0
        total_acc = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Create episode
            unique_labels = labels.unique()
            episode_labels = unique_labels[torch.randperm(len(unique_labels))[: self.n_way]]

            support_images = []
            support_labels = []
            query_images = []
            query_labels = []

            for idx, label in enumerate(episode_labels):
                mask = labels == label
                class_images = images[mask]
                perm = torch.randperm(len(class_images))

                # Select support and query samples
                support_idx = perm[: self.n_support]
                query_idx = perm[self.n_support : self.n_support + self.n_query]

                support_images.append(class_images[support_idx])
                support_labels.extend([idx] * self.n_support)
                query_images.append(class_images[query_idx])
                query_labels.extend([idx] * self.n_query)

            # Stack episodes
            support_images = torch.cat(support_images).to(self.device)
            support_labels = torch.tensor(support_labels).to(self.device)
            query_images = torch.cat(query_images).to(self.device)
            query_labels = torch.tensor(query_labels).to(self.device)

            # Get embeddings
            support_embeddings = self.model(support_images)
            query_embeddings = self.model(query_images)

            # Compute loss
            loss, loss_dict = self.loss_fn(
                support_embeddings,
                support_labels,
                query_embeddings,
                query_labels,
            )

            # Update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_acc += loss_dict["accuracy"]

        # Average metrics
        num_batches = len(train_loader)
        return {
            "total_loss": total_loss / num_batches,
            "accuracy": total_acc / num_batches,
        }

    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model using episodic evaluation"""
        self.model.eval()
        total_acc = 0
        episodes = 100  # Number of episodes to evaluate on

        with torch.no_grad():
            for episode in range(episodes):
                # Sample batch
                images, labels = next(iter(val_loader))

                # Create episode
                unique_labels = labels.unique()
                episode_labels = unique_labels[torch.randperm(len(unique_labels))[: self.n_way]]

                support_images = []
                support_labels = []
                query_images = []
                query_labels = []

                for idx, label in enumerate(episode_labels):
                    mask = labels == label
                    class_images = images[mask]
                    perm = torch.randperm(len(class_images))

                    support_idx = perm[: self.n_support]
                    query_idx = perm[self.n_support : self.n_support + self.n_query]

                    support_images.append(class_images[support_idx])
                    support_labels.extend([idx] * self.n_support)
                    query_images.append(class_images[query_idx])
                    query_labels.extend([idx] * self.n_query)

                # Stack episodes
                support_images = torch.cat(support_images).to(self.device)
                support_labels = torch.tensor(support_labels).to(self.device)
                query_images = torch.cat(query_images).to(self.device)
                query_labels = torch.tensor(query_labels).to(self.device)

                # Get embeddings
                support_embeddings = self.model(support_images)
                query_embeddings = self.model(query_images)

                # Compute accuracy
                _, loss_dict = self.loss_fn(
                    support_embeddings,
                    support_labels,
                    query_embeddings,
                    query_labels,
                )

                total_acc += loss_dict["accuracy"]

        return total_acc / episodes

    def train(
        self,
        data_module: ImagenetteDataModule,
        num_epochs: int = 100,
        episodes_per_epoch: int = 100,
        batch_size: int = 64,
    ) -> Dict[str, List[float]]:
        """Full training loop"""
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        history = {
            "total_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        for epoch in range(num_epochs):
            epoch_metrics = self.train_epoch(train_loader, data_module, epoch)

            # Record metrics
            history["total_loss"].append(epoch_metrics["total_loss"])
            history["train_accuracy"].append(epoch_metrics["accuracy"])

            self.writer.add_scalar("Loss/total", epoch_metrics["total_loss"], epoch)
            self.writer.add_scalar("Accuracy/train", epoch_metrics["accuracy"], epoch)

            # Validate every N epochs
            if (epoch + 1) % self.validation_freq == 0:
                val_accuracy = self.validate(val_loader)
                logger.info(f"Validation Mean Per Class Accuracy: {val_accuracy:.2f}%")
                history["val_accuracy"].append(val_accuracy)
                self.writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
                logger.info(f"Epoch {epoch+1}/{num_epochs}")
                logger.info(f"Total Loss: {epoch_metrics['total_loss']:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs}")
                logger.info(f"Total Loss: {epoch_metrics['total_loss']:.4f}")

        # Save model weights in the run directory
        os.makedirs(self.run_dir, exist_ok=True)
        save_path = os.path.join(self.run_dir, "last.pt")
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

        # Close tensorboard writer
        self.writer.close()
        return history
