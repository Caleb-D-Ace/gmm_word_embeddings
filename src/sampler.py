import torch

class NegativeSampler:
    """
    Creates a sample of negative context words for use when training.
    """
    def __init__(self, word_counts: list[int], power: float = 0.75):
        # Calculate the probability sampling for the whole vocabulary. 
        # This method is similar to the process in word2vec, where we decrease
        # the importance of word frequency for use as a negative context word
        # by raising it to the power of 0.75 (default).
        counts = torch.tensor(word_counts, dtype=torch.float32)
        adjusted_counts = counts ** power
        self.weights = adjusted_counts / torch.sum(adjusted_counts)

    def sample(self, num_samples: int) -> torch.Tensor:
        # Pick random word IDs using pytorch's multinomial method
        return torch.multinomial(self.weights, num_samples=num_samples, replacement=True)