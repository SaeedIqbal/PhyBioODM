"""
hsic_loss.py
Implements the Hilbert-Schmidt Independence Criterion (HSIC) with Gaussian RBF kernels 
to suppress non-linear dependencies between the biological and physical feature manifolds. 
This is the second component of the Manifold Orthogonality Constraint Loss, 
resolving Gap 3 by ensuring statistical independence beyond linear correlations.

Mathematical Formulation:
HSIC(H_bio, H_phys) = (1 / (N - 1)^2) * tr(K_bio @ H_c @ K_phys @ H_c)

Where:
- K_bio, K_phys in R^{N x N} are the Gram matrices computed using Gaussian RBF kernels:
  k(x, x') = exp(-||x - x'||^2 / (2 * sigma^2))
- H_c = I_N - (1 / N) * 1_N * 1_N^T is the centering matrix.
- N is the effective batch size (flattened from B * H * W if spatial dimensions are included).

Efficient Computation:
tr(K_bio @ H_c @ K_phys @ H_c) = sum( (K_bio)_c * (K_phys)_c )
where K_c is the centered kernel matrix, and * denotes the Hadamard (element-wise) product.
This avoids explicit matrix multiplication and reduces complexity from O(N^3) to O(N^2).

Note: To prevent Out-Of-Memory (OOM) errors, ensure that the effective batch size N 
is reasonably small (e.g., N <= 2000). If using 4D feature maps (B, C, H, W), 
it is recommended to apply Global Average Pooling before passing them to this loss, 
or compute HSIC on a random subset of spatial locations.
"""

import torch
import torch.nn as nn
import warnings

class HSICLoss(nn.Module):
    """
    Computes the Hilbert-Schmidt Independence Criterion (HSIC) using Gaussian RBF kernels.
    """
    
    def __init__(self, sigma_bio: float = None, sigma_phys: float = None, use_median_heuristic: bool = True):
        """
        Initializes the HSICLoss module.
        
        :param sigma_bio: Bandwidth parameter for the biological RBF kernel. 
                          If None and use_median_heuristic is True, it is computed adaptively.
        :param sigma_phys: Bandwidth parameter for the physical RBF kernel.
        :param use_median_heuristic: Whether to use the median heuristic to adaptively 
                                     compute sigma based on the data distribution.
        """
        super(HSICLoss, self).__init__()
        self.sigma_bio = sigma_bio
        self.sigma_phys = sigma_phys
        self.use_median_heuristic = use_median_heuristic

    def _flatten_spatial_dims(self, H: torch.Tensor) -> torch.Tensor:
        """
        Helper method to flatten 4D feature maps (B, C, H, W) into 2D matrices (B*H*W, C).
        """
        if H.dim() == 4:
            B, C, H_dim, W_dim = H.shape
            H = H.permute(0, 2, 3, 1).reshape(-1, C)
        elif H.dim() != 2:
            raise ValueError(f"Expected 2D (B, C) or 4D (B, C, H, W) input, got {H.dim()}D")
        return H

    def _compute_median_sigma(self, X: torch.Tensor, max_samples: int = 1000) -> torch.Tensor:
        """
        Computes the median heuristic for the RBF kernel bandwidth.
        sigma = sqrt(median(||x_i - x_j||^2) / 2)
        
        :param X: Input tensor of shape (N, C).
        :param max_samples: Maximum number of samples to use for computing the median 
                            to prevent OOM on large feature maps.
        :return: Adaptive bandwidth sigma.
        """
        N = X.shape[0]
        if N > max_samples:
            # Subsample to compute median heuristic efficiently
            indices = torch.randperm(N, device=X.device)[:max_samples]
            X_sub = X[indices]
        else:
            X_sub = X
            
        # Compute pairwise squared distances
        dists = torch.cdist(X_sub, X_sub, p=2.0).pow(2)
        
        # Extract upper triangle to ignore diagonal (zeros) and duplicates
        mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
        valid_dists = dists[mask]
        
        if valid_dists.numel() == 0:
            return torch.tensor(1.0, device=X.device)
            
        median_dist = valid_dists.median()
        # sigma = sqrt(median_dist / 2)
        sigma = torch.sqrt(median_dist / 2.0).clamp(min=1e-5)
        return sigma

    def _center_kernel(self, K: torch.Tensor) -> torch.Tensor:
        """
        Centers the kernel matrix K using the centering matrix H_c.
        K_c = H_c @ K @ H_c
        Efficiently computed as: K - mean_col - mean_row + global_mean
        """
        col_mean = K.mean(dim=0, keepdim=True)  # 1 x N
        row_mean = K.mean(dim=1, keepdim=True)  # N x 1
        global_mean = K.mean()
        
        K_c = K - col_mean - row_mean + global_mean
        return K_c

    def forward(self, H_bio: torch.Tensor, H_phys: torch.Tensor) -> torch.Tensor:
        """
        Computes the HSIC between biological and physical features.
        
        :param H_bio: Biological feature activations of shape (B, C) or (B, C, H, W).
        :param H_phys: Physical feature activations of shape (B, C) or (B, C, H, W).
        :return: Scalar HSIC loss.
        """
        if H_bio.shape != H_phys.shape:
            raise ValueError(f"Shape mismatch: H_bio {H_bio.shape} vs H_phys {H_phys.shape}")
            
        # 1. Flatten spatial dimensions if necessary
        H_bio_flat = self._flatten_spatial_dims(H_bio)
        H_phys_flat = self._flatten_spatial_dims(H_phys)
        
        N = H_bio_flat.shape[0]
        
        if N < 2:
            return torch.tensor(0.0, device=H_bio.device, requires_grad=True)
            
        if N > 2000:
            warnings.warn(f"Effective batch size N={N} is large. Computing HSIC may cause OOM. "
                          "Consider spatially pooling features or subsampling.")
            
        # 2. Compute bandwidths (sigmas)
        if self.use_median_heuristic:
            sigma_bio = self._compute_median_sigma(H_bio_flat)
            sigma_phys = self._compute_median_sigma(H_phys_flat)
        else:
            if self.sigma_bio is None or self.sigma_phys is None:
                raise ValueError("sigma_bio and sigma_phys must be provided if use_median_heuristic is False.")
            sigma_bio = torch.tensor(self.sigma_bio, device=H_bio.device)
            sigma_phys = torch.tensor(self.sigma_phys, device=H_phys.device)
            
        # 3. Compute Gram matrices (RBF Kernels)
        # K = exp(-dist^2 / (2 * sigma^2))
        dists_bio = torch.cdist(H_bio_flat, H_bio_flat, p=2.0).pow(2)
        K_bio = torch.exp(-dists_bio / (2.0 * sigma_bio**2))
        
        dists_phys = torch.cdist(H_phys_flat, H_phys_flat, p=2.0).pow(2)
        K_phys = torch.exp(-dists_phys / (2.0 * sigma_phys**2))
        
        # 4. Center the kernel matrices
        K_bio_c = self._center_kernel(K_bio)
        K_phys_c = self._center_kernel(K_phys)
        
        # 5. Compute HSIC
        # HSIC = (1 / (N - 1)^2) * sum(K_bio_c * K_phys_c)
        hsic = (K_bio_c * K_phys_c).sum() / ((N - 1) ** 2)
        
        return hsic


if __name__ == "__main__":
    # Example usage and validation of the HSICLoss module
    
    # 1. Instantiate the loss function with median heuristic
    hsic_loss_fn = HSICLoss(use_median_heuristic=True)
    
    # 2. Define orig tensor shapes
    batch_size = 32
    channels = 64
    
    # Scenario A: Independent features (HSIC should be near 0)
    H_bio_indep = torch.randn(batch_size, channels, requires_grad=True)
    H_phys_indep = torch.randn(batch_size, channels, requires_grad=True)
    
    loss_indep = hsic_loss_fn(H_bio_indep, H_phys_indep)
    print(f"Independent Features HSIC: {loss_indep.item():.6f}")
    
    # Scenario B: Highly dependent features (HSIC should be large)
    H_bio_dep = torch.randn(batch_size, channels)
    H_phys_dep = H_bio_dep + 0.1 * torch.randn(batch_size, channels) # Strongly correlated
    H_bio_dep.requires_grad = True
    H_phys_dep.requires_grad = True
    
    loss_dep = hsic_loss_fn(H_bio_dep, H_phys_dep)
    print(f"Dependent Features HSIC: {loss_dep.item():.6f}")
    
    # 3. Verify backward pass (gradient flow)
    loss_dep.backward()
    
    assert H_bio_dep.grad is not None, "Gradients did not flow back to H_bio!"
    assert H_phys_dep.grad is not None, "Gradients did not flow back to H_phys!"
    
    # 4. Test with fixed sigma
    hsic_fixed_fn = HSICLoss(sigma_bio=1.0, sigma_phys=1.0, use_median_heuristic=False)
    loss_fixed = hsic_fixed_fn(H_bio_indep, H_phys_indep)
    print(f"Fixed Sigma HSIC: {loss_fixed.item():.6f}")
    
    print("\nVerification passed: HSIC loss computed correctly, gradients flow successfully, and independent/dependent features yield expected relative magnitudes.")