import sys
import argparse
import logging
import debugpy
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)

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
    parser.add_argument(
        "--debug", action="store_true", help="Wait for debugpy connection on port 5678"
    )
    parser.add_argument("--exp-name", type=str, default="", help="Name of experiment.")

    args = parser.parse_args()

    if args.debug:
        logger.info("Waiting for debugpy connection on port 5678...")
        debugpy.listen(5678)
        debugpy.wait_for_client()
        logger.info("Debugpy client connected!")

    # Load dataset
    data = ImagenetteDataModule(args.data_path, batch_size=args.batch_size)

    # Train for each ratio
    for ratio in args.labeled_ratio:
        logger.info(
            f"\nStarting training for experiment '{args.exp_name}' with labeled_ratio: {ratio}"
        )
        trainer = SemiSupervisedTrainer(
            num_classes=10, labeled_ratio=ratio, exp_name=args.exp_name
        )
        history = trainer.train(
            data, num_epochs=args.epochs, batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
