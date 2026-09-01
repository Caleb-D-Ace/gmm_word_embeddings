import math
import torch
import torch.nn as nn
import preprocess

"""
PtModel contains the mathematical functions and PyTorch nn.module architecture.
"""
class PtModel():
    
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

    def gmm_energy(log_overlap_matrix, p1, p2):
        """
        Loss function: Step 2
        Computes the GMM energy for each batch item based on the log-overlap matrix and mixture weights.

        Math:
            E(f, g) = log(Σ_k Σ_m [ w_k * w_m * exp(log_Δ(f_k, g_m)) ])
        Args:
            log_overlap_matrix (Tensor): Pairwise component log-overlaps of shape (batch_size, K, K)
            p1 (Tensor): Target word mixture weights of shape (batch_size, K)
            p2 (Tensor): Context word mixture weights of shape (batch_size, K)
        """

        # Get the combined mixture weights for each pair of components
        p1_expand = p1.unsqueeze(2)   # Shape: (batch_size, K, 1)
        p2_expand = p2.unsqueeze(1)   # Shape: (batch_size, 1, K)
        mix_weights = p1_expand * p2_expand  # Shape: (batch_size, K, K)

        # Convert log-overlap from log-space to probability space so we can do calculations with it
        overlap_prob = torch.exp(log_overlap_matrix)

        # Calculate the weighted sum of overlaps
        weighted_sum = mix_weights * overlap_prob  # Shape: (batch_size, K, K)

        # Sum across both K axes 
        sum_over_components = torch.sum(weighted_sum, dim=(-2, -1))  # Shape: (batch_size,) <- this is the final energy score for each batch item

        # Return the log of the sum to get the final GMM energy in log space
        return torch.log(sum_over_components)  # Shape: (batch_size,)

    def max_margin_ranking(word, context, negative_context):
        """
        Loss function: Step 3
        Computes the max-margin ranking loss for a given word, its context, and negative context words.
        Because our GMM energy function returns a result in log-space, we can directly use it to compute the loss.

        Math:
            L = max(0, margin - E(word, context) + E(word, negative_context))
        """