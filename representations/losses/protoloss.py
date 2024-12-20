import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from typing import Tuple, Dict


logger = logging.getLogger(__name__)


class ProtoLoss(nn.Module):
    """
    Prototypical Network Loss
    """

    def __init__(
        self,
        writer: SummaryWriter,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.writer = writer
        self.temperature = temperature

    def forward(
        self,
        features_labeled: Tensor,
        support_labels: Tensor,
        features_unlabeled: Tensor,
        query_labels: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute prototypical network loss

        Args:
            features_labeled: Support set embeddings [N_s, D]
            support_labels: Support set labels [N_s]
            features_unlabeled: Query set embeddings [N_q, D]
            query_labels: Query set labels [N_q]

        Returns:
            loss: Prototypical network loss
            loss_dict: Dictionary with loss components
        """
        # Compute prototypes for each class
        prototypes = []
        classes = torch.unique(support_labels)

        for c in classes:
            # Get embeddings for class c
            mask = support_labels == c
            class_embeddings = features_labeled[mask]
            # Compute prototype (mean embedding)
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)

        # Compute distances between queries and prototypes
        dists = torch.cdist(features_unlabeled, prototypes, p=2)

        # Convert distances to probabilities with temperature scaling
        logits = -dists / self.temperature
        log_probs = F.log_softmax(logits, dim=1)

        # Convert query labels to indices
        query_indices = torch.zeros_like(query_labels)
        for idx, c in enumerate(classes):
            query_indices[query_labels == c] = idx

        # Compute cross entropy loss
        loss = F.nll_loss(log_probs, query_indices)

        # Compute accuracy
        pred = log_probs.argmax(dim=1)
        accuracy = (pred == query_indices).float().mean()

        loss_dict = {"proto_loss": loss.item(), "accuracy": accuracy.item()}

        return loss, loss_dict
