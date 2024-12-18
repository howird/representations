import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


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
        self.strong_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.08, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
                transforms.RandomAutocontrast(p=0.2),
                transforms.RandomEqualize(p=0.2),
                transforms.RandomPosterize(bits=4, p=0.2),
                transforms.RandomSolarize(threshold=128, p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
                transforms.RandomErasing(p=0.1),
            ]
        )

        # Weak augmentation for consistency training
        self.weak_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),  # Less aggressive crop
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),  # Milder color jittering
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        # Standard transforms for labeled data
        self.train_transforms = self.strong_transforms

        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(
                    int(image_size * 1.15), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
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
