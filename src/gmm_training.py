from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from dataset import SkipGramDataset
from gmm_word_embedding import GMMWordEmbedding
from sampler import NegativeSampler


@dataclass
class TrainingConfig:
    processed_dir: str = "data/processed"
    output_path: str = "data/model/gmm_embeddings.npz"
    embedding_dim: int = 50
    K: int = 2
    window_size: int = 5
    batch_size: int = 256
    epochs: int = 5
    lr: float = 1e-3
    margin: float = 1.0
    num_negatives: int = 5


"""
GmmTrainer runs through the .memmap file created by gmm_word_embedding.py, passes its data into PyTorch, and trains a probablistic
word embedding model using the energy math from gmm_model.py.
It outputs a .npz file containing the fully-trained model.
"""
class GmmTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        # 2. Load vocabulary and metadata

        # 3. Instantiate pipelines

        # 4. Training loop

        # 5. Save the trained model
        pass