import math

import torch

import preprocess

"""
PtModel contains the mathematical functions and PyTorch nn.module architecture.
"""
class PtModel():
    def loss_function():
        return
    def elk_function():
        return

    
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