import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
gmm_word_embedding contains the GMM energy function implementation for training the word embedding model.
It uses the PyTorch.nn module to create a neural network trained using this energy function. 
The model is trained using a max-margin ranking loss function, which is implemented in the max_margin_ranking method.
"""
class GMMWordEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 50, K: int = 2):
        super().__init__()
        # Set up the basic parameters for the equation. These are used in the log_overlap and gmm_energy functions.
        self.vocab_size = vocab_size    # Size of the vocabulary (number of unique words)
        self.D = embedding_dim          # Dimension of the embedding space
        self.K = K                      # How many modes there are

        # Set up the embedding layers for the means, variances, and mixture weights. These are trainable parameters.
        self.mu_embeddings = nn.Embedding(vocab_size, K * self.D)
        self.var_embeddings = nn.Embedding(vocab_size, K * self.D)
        self.mix_logits = nn.Embedding(vocab_size, K)

        # Initialize weights with standard distributions
        nn.init.uniform_(self.mu_embeddings.weight, -0.5 / self.D, 0.5 / self.D)
        nn.init.uniform_(self.var_embeddings.weight, 0.1, 1.0) 
        nn.init.zeros_(self.mix_logits.weight)

    def get_word_params(self, word_ids: torch.Tensor):
        """
        Given a batch of word ids, this method retrieves the corresponding means, variances, and mixture weights.
        It reshapes the means and variances to have shape (batch_size, K, D) for further calculations.
        """
        batch_size = word_ids.shape[0]

        # Get mu and var for word_ids as a 1D representation
        mu = self.mu_embeddings(word_ids)
        var = self.var_embeddings(word_ids)

        # Ensure variance is strictly positive using Exponential or Softplus
        var = F.softplus(var) + 1e-4

        # Reshape from (batch_size, K*D) to (batch_size, K, D)
        mu = mu.view(batch_size, self.K, self.D)
        var = var.view(batch_size, self.K, self.D)

        # Use softmax to convert logits to probabilities for mixture weights
        mix_logits = self.mix_logits(word_ids)
        mix_weights = torch.softmax(mix_logits, dim=-1)  # Convert logits to probabilities

        return mu, var, mix_weights
    
    @staticmethod
    def log_overlap(mu1, mu2, var1, var2):
        """
        Loss function: Step 1
        Computes the closed-form Expected Likelihood Kernel (log-overlap) between 
        Gaussian components across D dimensions for all batch items and component pairs.

        Math:
            log_Δ(f_k, g_m) = -0.5 * Σ_d [ log(var1 + var2) + (mu1 - mu2)^2 / (var1 + var2) ] - (D / 2) * log(2π)

        Args:
            mu1 (Tensor): Target word means of shape (batch_size, K, D)
            mu2 (Tensor): Context word means of shape (batch_size, K, D)
            var1 (Tensor): Target word variances (σ²) of shape (batch_size, K, D)
            var2 (Tensor): Context word variances (σ²) of shape (batch_size, K, D)

        Returns:
            Tensor: Pairwise component log-overlaps of shape (batch_size, K, K)
        """
        # Expand the dimensions for our input tensors for calculation
        mu1_expand = mu1.unsqueeze(2)   # Shape: (batch_size, K, 1, D)
        var1_expand = var1.unsqueeze(2) # Shape: (batch_size, K, 1, D)
        mu2_expand = mu2.unsqueeze(1)   # Shape: (batch_size, 1, K, D)
        var2_expand = var2.unsqueeze(1) # Shape: (batch_size, 1, K, D)

        # Get vector dimension D dynamically from the last axis
        D = mu1.shape[-1]

        # Calculate combined variance
        var_combined = var1_expand + var2_expand    # Shape: (batch_size, K, K, D)

        # Calculate normalized distance
        norm_dist = ((mu1_expand - mu2_expand)**2) / var_combined

        # Calculate combined variance penalty
        var_penalty = torch.log(var_combined)

        # Inner equation
        overlap = var_penalty + norm_dist

        # Sum across D dimensions (you can find the Dth dimension using dim=-1)
        dimensional_sum = torch.sum(overlap, dim=-1)

        log_constant = math.log(2 * math.pi)

        return -0.5 * dimensional_sum - (0.5 * D * log_constant)

    @staticmethod
    def gmm_energy(log_overlap_matrix, p1, p2):
        """
        Loss function: Step 2
        Computes the GMM energy for each batch item based on the log-overlap matrix and mixture weights.

        Math:
            E(f, g) = log(Σ_k Σ_m [ w_k * w_m * exp(log_Δ(f_k, g_m)) ])
        Computed in log space with logsumexp, because exp() of typical log-overlaps (around -60 at D=50) underflows.
        Args:
            log_overlap_matrix (Tensor): Pairwise component log-overlaps of shape (batch_size, K, K)
            p1 (Tensor): Target word mixture weights of shape (batch_size, K)
            p2 (Tensor): Context word mixture weights of shape (batch_size, K)
        """

        # Log of the combined mixture weight for each pair of components: log(w_k * w_m) = log(w_k) + log(w_m)
        log_p1_expand = torch.log(p1).unsqueeze(2)   # Shape: (batch_size, K, 1)
        log_p2_expand = torch.log(p2).unsqueeze(1)   # Shape: (batch_size, 1, K)
        log_mix_weights = log_p1_expand + log_p2_expand  # Shape: (batch_size, K, K)

        # log(Σ w_k * w_m * exp(log_Δ)) == logsumexp(log(w_k * w_m) + log_Δ), summed across both K axes
        return torch.logsumexp(log_mix_weights + log_overlap_matrix, dim=(-2, -1))  # Shape: (batch_size,)


    def forward(self, target_ids: torch.Tensor, ctx_ids: torch.Tensor):
        """
        Forward pass execution method for PyTorch.
        Takes a batch of target and context word ids and returns the energy score for the whole batch.
        """
        # Look up target word parameters (means, variances, mixture weights)
        mu1, var1, p1 = self.get_word_params(target_ids)
        mu2, var2, p2 = self.get_word_params(ctx_ids)

        # Create log overlap matrix
        overlap = self.log_overlap(mu1, mu2, var1, var2)

        # Calculate total GMM energy based on the overlaps and the probability weights
        gmm_energy = self.gmm_energy(overlap, p1, p2)

        return gmm_energy


    @staticmethod
    def max_margin_ranking(E_pos: torch.Tensor, E_neg: torch.Tensor, margin: float = 1.0):
        """
        Loss function: Step 3
        Computes hinged margin ranking loss on paired energies. 
        This ensures that backpropogation pulls positive pairs closer together and pushes negative pairs away.

        Math:
            L = max(0, margin - E(word, context) + E(word, negative_context))
        """
        # margin - E_pos + E_neg
        losses = margin - E_pos + E_neg

        hinge_loss = torch.clamp(losses, min=0.0)

        return torch.mean(hinge_loss)
