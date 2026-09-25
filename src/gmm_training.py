from dataclasses import dataclass

import json
from pathlib import Path
import logging
import numpy as np
import time
import torch
from torch.utils.data import DataLoader

from dataset import SkipGramDataset
from gmm_word_embedding import GMMWordEmbedding
import metrics_logger
from sampler import NegativeSampler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class TrainingConfig:
    processed_dir: str = "data/processed"
    output_path: str = "data/model"
    log_dir: str = "logs/"
    embedding_dim: int = 50
    K: int = 2
    window_size: int = 5
    batch_size: int = 256
    epochs: int = 5
    lr: float = 1e-3
    margin: float = 1.0
    num_negatives: int = 1  # DO NOT CHANGE THIS VALUE. Code to support multiple negative samples is not yet implemented.
    checkpoint_every: int = 0  # 0 disables checkpointing; e.g. 10 saves a snapshot every 10 epochs into output_path/epoch_<N>/

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
        sampler = NegativeSampler(word_counts=word_count_list)

        # Create DataLoader for batching and optimizer
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(embedding_model.parameters(), lr=self.config.lr)

        # 4. Training loop
        # Create MetricsLogger object for logging
        with metrics_logger.MetricsLogger(self.config.log_dir) as metrics:
            for epoch in range(self.config.epochs):
                # Set starting epoch timestamp
                epoch_start_time = time.perf_counter()
                # Loss is a per-batch mean, so weight each batch by its pair count to get an exact epoch average
                # even when the last batch is smaller.
                epoch_loss_sum = 0.0
                epoch_pair_count = 0

                for batch_idx, (centers, contexts) in enumerate(dataloader):
                    # Set starting batch timestamp
                    batch_start_time = time.perf_counter()

                    centers = centers.to(self.device)
                    contexts = contexts.to(self.device)

                    # Reshape the output of SkipGramDataset to match input shape of (target, context)
                    target_ids, ctx_ids = self.reshape(centers, contexts)
                    # Get our negative samples for this batch
                    negative_samples = sampler.sample(self.config.num_negatives * ctx_ids.size(0)).to(self.device)

                    # Get positive and negative energies for this batch
                    e_pos = embedding_model.forward(target_ids, ctx_ids)
                    e_neg = embedding_model.forward(target_ids, negative_samples)

                    # Compute the max-margin ranking loss
                    loss = embedding_model.max_margin_ranking(e_pos, e_neg, self.config.margin)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # Calculate time and other metric parameters
                    loss_item = loss.item()
                    epoch_loss_sum += loss_item * e_pos.numel()
                    epoch_pair_count += e_pos.numel()
                    e_pos_mean = e_pos.mean().item()
                    e_neg_mean = e_neg.mean().item()
                    batch_elapsed_sec = time.perf_counter() - batch_start_time
                    epoch_elapsed_sec = time.perf_counter() - epoch_start_time # This aggregates over each batch until we finish the epoch.

                    metrics.writeRow(
                        epoch=epoch, batch=batch_idx, epoch_sec=epoch_elapsed_sec,
                        batch_sec=batch_elapsed_sec, loss=loss_item, pos_energy=e_pos_mean,
                        neg_energy=e_neg_mean, active_pair=0, grad_norm=0,
                        mix_weight=0, var_min=0, var_max=0, var_mean=0
                    )

                # Periodic checkpoint: Saves a model after every N epochs
                if self.config.checkpoint_every and (epoch + 1) % self.config.checkpoint_every == 0:
                    self.save_model(vocab, vocab_size, embedding_model, subdir=f"epoch_{epoch + 1}")
                logger.info(f"Epoch {epoch + 1}/{self.config.epochs} completed. Avg loss: {epoch_loss_sum / epoch_pair_count:.4f}")

        # 5. Save the final trained model
        self.save_model(vocab, vocab_size, embedding_model)

    """
    Method to save the trained GMM embeddings and the vocabulary to disk. The embeddings are saved in both .npz and .pt formats.
    Parameters:
        subdir - optional subdirectory name under output_path (e.g. "epoch_10") for a mid-training checkpoint;
                 omitted for the final save, which writes directly into output_path as before.
    """
    def save_model(self, vocab, vocab_size, embedding_model, subdir: str = None):
        output_dir = Path(self.config.output_path)
        if subdir is not None:
            output_dir = output_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        # get_word_params applies the same softplus/softmax transforms used during training,
        # so the saved arrays are ready-to-use variances/mixture weights, not raw logits.
        with torch.no_grad():
            all_ids = torch.arange(vocab_size, device=self.device)
            mu, var, mix_weights = embedding_model.get_word_params(all_ids)

        np.savez(
            output_dir / "gmm_embeddings.npz",
            mu=mu.cpu().numpy(),
            var=var.cpu().numpy(),
            mix_weights=mix_weights.cpu().numpy(),
        )
        torch.save(embedding_model.state_dict(), output_dir / "gmm_embeddings.pt")

        # Keep a copy of the vocab alongside the model it was actually trained on, so the two
        # never drift apart even if data/processed/ gets re-preprocessed later.
        with open(output_dir / "sorted_vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

    @staticmethod
    def reshape(centers: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        """
        Reshapes the center/context word embeddings from SkipGramDataset to match the expected input shape for the GMMWordEmbedding model.
        Parameters:
            centers - Tensor of shape (batch_size,)
            contexts - Tensor of shape (batch_size, 2*window_size)
        Returns:
            reshaped_tensor - Tensor of shape (2, batch_size * 2*window_size), meant to be unpacked as
                target_ids, ctx_ids = reshape(centers, contexts)
                each of shape (batch_size * 2*window_size,), aligned so target_ids[i] is paired with ctx_ids[i]
        """
        repeated_centers = torch.repeat_interleave(centers, contexts.size(1))
        flat_ctx = contexts.flatten()
        reshaped_tensor = torch.stack((repeated_centers, flat_ctx), dim=0)
        return reshaped_tensor