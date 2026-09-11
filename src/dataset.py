import numpy as np
import torch
from torch.utils.data import Dataset
"""
SkipGramDataset uses .memmap to read an indexed corpus .bin to get center and ctx words from our corpus.
"""
class SkipGramDataset(Dataset):
    def __init__(self, bin_file: str, window_size: int = 5, dtype = np.uint16):
        # memmap allows us to read the corpus without loading it all into RAM
        self.data = np.memmap(bin_file, dtype = dtype, mode = 'r')
        self.window_size = window_size
        self.data_len = len(self.data)

    def __len__(self):
        # Adjust the length by the size of the window, since we don't want to get an index out-of-bounds error.
        return self.data_len - (2 * self.window_size)

    def __getitem__(self, idx):
        # Get center word
        center_idx = idx + self.window_size
        center_word = self.data[center_idx]
        
        # Get context words
        left_bound = idx
        right_bound = center_idx + self.window_size

        left_ctx = self.data[left_bound : center_idx]
        right_ctx = self.data[center_idx + 1 : right_bound]

        ctx_words = np.concatenate([left_ctx, right_ctx])
        return torch.tensor(center_word, dtype=torch.long), torch.tensor(ctx_words, dtype=torch.long)