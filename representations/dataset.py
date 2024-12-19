import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import InterpolationMode, v2


class ImagenetteDataModule:
    """
    DataModule for the Imagenette dataset (subset of ImageNet)
    Applies standard ImageNet-style augmentations
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        image_size: int = 224,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.pin_memory = pin_memory

        # ImageNet statistics
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # Strong augmentation for consistency training
        self.strong_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.RandomResizedCrop(
                    size=self.image_size,
                    scale=(0.8, 1.0),  # default
                    antialias=True,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                v2.RandomAutocontrast(p=0.2),
                v2.RandomEqualize(p=0.2),
                v2.RandomPosterize(bits=4, p=0.2),
                v2.RandomSolarize(threshold=128, p=0.1),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std),
                v2.RandomErasing(p=0.1),
            ]
        )

        # Weak augmentation for consistency training
        self.weak_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.RandomResizedCrop(
                    size=self.image_size,
                    scale=(0.8, 1.0),
                    antialias=True,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),  # Milder color jittering
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )

        # Standard v2 for labeled data
        self.train_transforms = self.strong_transforms

        self.val_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(
                    int(image_size * 1.15), interpolation=InterpolationMode.BICUBIC
                ),
                v2.CenterCrop(self.image_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def train_dataloader(self, transform=None) -> DataLoader:
        """Get training dataloader"""
        dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, "train"),
            transform=transform if transform is not None else self.train_transforms,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader"""
        dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, "val"), transform=self.val_transforms
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
