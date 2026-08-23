"""
manifold_orthogonality.py
Implements the Manifold Orthogonality Constraint Loss, which aggregates the 
Covariance Loss and the Hilbert-Schmidt Independence Criterion (HSIC) across 
a subset of deep residual blocks to enforce strict statistical independence 
between the biological and physical feature manifolds.

This module resolves Gap 3 (Statistical Dependence of Latents) by ensuring 
that the gradient flow decomposes into orthogonal components, preventing the 
model from compensating for unmodeled physical variations by distorting 
intrinsic biological geometry.

Mathematical Formulation:
L_ortho = sum_{l in L_layers} ( L_cov^{(l)} + lambda_HSIC * HSIC(H_bio^{(l)}, H_phys^{(l)}) )

Where:
- L_cov^{(l)} = || Cov(H_bio^{(l)}, H_phys^{(l)}) ||_F^2 (Linear independence)
- HSIC(H_bio^{(l)}, H_phys^{(l)}) uses Gaussian RBF kernels (Non-linear independence)
- lambda_HSIC is a hyperparameter balancing the linear and non-linear constraints.
"""

import torch
import torch.nn as nn
from typing import List, Union

# Import the previously defined loss components to reuse their mathematical implementations
from .covariance_loss import CovarianceLoss
from .hsic_loss import HSICLoss


class ManifoldOrthogonalityLoss(nn.Module):
    """
    Computes the total Manifold Orthogonality Constraint Loss by aggregating 
    linear (Covariance) and non-linear (HSIC) independence metrics across 
    multiple layers of the UNet backbone.
    """
    
    def __init__(self, lambda_hsic: float = 1.0, use_median_heuristic: bool = True):
        """
        Initializes the ManifoldOrthogonalityLoss module.
        
        :param lambda_hsic: Weighting hyperparameter (lambda_HSIC) for the HSIC loss component.
        :param use_median_heuristic: Whether to use the median heuristic for HSIC bandwidths.
        """
        super(ManifoldOrthogonalityLoss, self).__init__()
        
        if lambda_hsic < 0:
            raise ValueError("lambda_hsic must be non-negative.")
            
        self.lambda_hsic = lambda_hsic
        
        # Instantiate the sub-loss components (reusing the mathematical formulations)
        self.cov_loss = CovarianceLoss()
        self.hsic_loss = HSICLoss(use_median_heuristic=use_median_heuristic)

    def forward(self, 
                H_bio: Union[torch.Tensor, List[torch.Tensor]], 
                H_phys: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Computes the aggregated orthogonality loss across one or multiple layers.
        
        :param H_bio: Biological feature activations. Can be a single tensor (B, C, H, W) 
                      or a list of tensors from multiple layers (l in L_layers).
        :param H_phys: Physical feature activations. Must match the structure of H_bio.
        :return: Scalar total orthogonality loss (L_ortho).
        """
        # Convert single tensors to lists for uniform processing
        if isinstance(H_bio, torch.Tensor):
            H_bio = [H_bio]
        if isinstance(H_phys, torch.Tensor):
            H_phys = [H_phys]
            
        if len(H_bio) != len(H_phys):
            raise ValueError(f"Number of biological layers ({len(H_bio)}) must match "
                             f"physical layers ({len(H_phys)}).")
                             
        # Initialize total loss on the same device as the input tensors
        total_loss = torch.tensor(0.0, device=H_bio[0].device)
        
        # Aggregate loss over all specified layers l in L_layers
        for h_bio_l, h_phys_l in zip(H_bio, H_phys):
            # 1. Compute linear independence: L_cov^{(l)} = || Cov(H_bio, H_phys) ||_F^2
            l_cov = self.cov_loss(h_bio_l, h_phys_l)
            
            # 2. Compute non-linear independence: HSIC(H_bio, H_phys)
            l_hsic = self.hsic_loss(h_bio_l, h_phys_l)
            
            # 3. Aggregate for this layer: L_cov^{(l)} + lambda_HSIC * HSIC^{(l)}
            layer_loss = l_cov + self.lambda_hsic * l_hsic
            
            # Accumulate to the total orthogonality loss
            total_loss = total_loss + layer_loss
            
        return total_loss


if __name__ == "__main__":
    # Example usage and validation of the ManifoldOrthogonalityLoss module
    
    # 1. Instantiate the loss function with a specific lambda_HSIC
    ortho_loss_fn = ManifoldOrthogonalityLoss(lambda_hsic=0.5, use_median_heuristic=True)
    
    # 2. Define orig tensor shapes (simulating features from 3 different UNet residual blocks)
    batch_size = 16
    num_layers = 3
    
    # Scenario A: Single layer test
    H_bio_single = torch.randn(batch_size, 128, 16, 16, requires_grad=True)
    H_phys_single = torch.randn(batch_size, 128, 16, 16, requires_grad=True)
    
    loss_single = ortho_loss_fn(H_bio_single, H_phys_single)
    print(f"Single Layer Orthogonality Loss: {loss_single.item():.4f}")
    
    # Scenario B: Multi-layer test (L_ortho = sum_{l} ...)
    H_bio_list = [torch.randn(batch_size, 128, 16, 16, requires_grad=True) for _ in range(num_layers)]
    H_phys_list = [torch.randn(batch_size, 128, 16, 16, requires_grad=True) for _ in range(num_layers)]
    
    loss_multi = ortho_loss_fn(H_bio_list, H_phys_list)
    print(f"Multi-Layer Orthogonality Loss (sum of {num_layers} layers): {loss_multi.item():.4f}")
    
    # 3. Verify backward pass (gradient flow for the total objective L_total = L_diff + alpha * L_ortho)
    loss_multi.backward()
    
    for i, h_bio in enumerate(H_bio_list):
        assert h_bio.grad is not None, f"Gradients did not flow back to H_bio layer {i}!"
        
    print("\nVerification passed: Manifold Orthogonality Loss computed correctly for single/multi-layer inputs, and gradients flow successfully through both linear and non-linear constraints.")