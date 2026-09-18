"""
This is the main entry point for training the GMM word embedding model.
It expects data/processed/ to already exist (run the preprocess command on a corpus first).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from gmm_training import GmmTrainer, TrainingConfig


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train GMM Word Embeddings")

    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Directory containing the preprocessed corpus (default: data/processed)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/model",
        help="Where to save the trained model (default: data/model/gmm_embeddings.[filetype])\n\tFiles are saved in both a .npz and a .pt format."
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=50,
        help="Dimensionality of each Gaussian component (default: 50)"
    )
    parser.add_argument(
        "--k", dest="K",
        type=int,
        default=2,
        help="Number of mixture components per word (default: 2)"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Context window size on each side of the center word (default: 5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training (default: 256)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Margin for the max-margin ranking loss (default: 1.0)"
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=1,
        help="Negative samples drawn per positive pair (default: 5)"
    )

    args = parser.parse_args()
    return TrainingConfig(**vars(args))


def main():
    config = parse_args()
    trainer = GmmTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
