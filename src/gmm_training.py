from dataclasses import dataclass

import json
from pathlib import Path
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
        # Verify that the processed directory contains needed files
        if not Path(self.config.processed_dir).exists():
            raise FileNotFoundError(f"Processed directory '{self.config.processed_dir}' does not exist. Please run the preprocess command first.")
        if not Path(self.config.processed_dir, "sorted_vocab.json").exists():
            raise FileNotFoundError(f"Vocabulary file 'sorted_vocab.json' not found in '{self.config.processed_dir}'. Please run the preprocess command first.")
        if not Path(self.config.processed_dir, "corpus_index.bin").exists():
            raise FileNotFoundError(f"Corpus index file 'corpus_index.bin' not found in '{self.config.processed_dir}'. Please run the preprocess command first.")
        
        with open(Path(self.config.processed_dir) / "sorted_vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
            vocab_size = len(vocab)
        word_count_list = [count for rank, count in sorted(vocab.values(), key=lambda x: x[0])]
        
        # 3. Instantiate pipelines
        embedding_model = GMMWordEmbedding(vocab_size, self.config.embedding_dim, self.config.K).to(self.device)
        dataset = SkipGramDataset(bin_file=Path(self.config.processed_dir) / "corpus_index.bin", window_size=self.config.window_size)
        sampler = NegativeSampler(vocab_size, word_counts=word_count_list)

        # 4. Training loop

        # 5. Save the trained model
        pass