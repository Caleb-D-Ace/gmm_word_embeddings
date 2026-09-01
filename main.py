"""
This is the main entry point for the training model.
Instructions:
When processing the corpus to build a vocabulary, words with a frequency count below a given threshold are cut from our final vocabulary. 
This removes typos as well as very rare words that may not have enough context information to create good mappings. By default, this threshold is set to 40.
Press "Enter" to keep the default. Otherwise, type a new threshold and press "Enter".
"""
import argparse
import dataset
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description="Train GMM Word Embeddings")
    
    # Default batch size is 256. This can be overridden by the user via command line argument.
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=256, 
        help="Batch size for training (default: 256)"
    )
    
    args = parser.parse_args()

    # Pass the dynamic batch_size into your DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )