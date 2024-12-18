import sys
import os
import argparse
from typing import List

sys.path.append("../representations")

from representations.dataset import ImagenetteDataModule
from representations.trainer import SemiSupervisedTrainer


def parse_ratio(ratio_str: str) -> List[float]:
    """Parse comma-separated ratios or single ratio"""
    try:
        return [float(x.strip()) for x in ratio_str.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("Ratios must be floating point numbers")


def main():
    parser = argparse.ArgumentParser(
        description="Train semi-supervised model on Imagenette"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to Imagenette dataset"
    )
    parser.add_argument(
        "--labeled_ratio",
        type=parse_ratio,
        required=True,
        help="Ratio of labeled data (single float or comma-separated list)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Number of samples in a batch"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )

    args = parser.parse_args()

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Load dataset
    data = ImagenetteDataModule(args.data_path, batch_size=args.batch_size)

    # Train for each ratio
    for ratio in args.labeled_ratio:
        print(f"\nTraining with labeled_ratio: {ratio}")
        trainer = SemiSupervisedTrainer(num_classes=10, labeled_ratio=ratio)
        history = trainer.train(data, num_epochs=args.epochs)


if __name__ == "__main__":
    main()
