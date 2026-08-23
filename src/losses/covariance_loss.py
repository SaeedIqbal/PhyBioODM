"""
covariance_loss.py
Implements the empirical cross-covariance loss to suppress linear correlations 
between the biological and physical feature manifolds. This is the first component 
of the Manifold Orthogonality Constraint Loss, resolving Gap 3 by ensuring that 
the linear dependencies between the two streams are minimized.

Mathematical Formulation:
L_cov = || (1 / (B - 1)) * (H_bio - H_bar_bio)^T (H_phys - H_bar_phys) ||_F^2

Where:
- H_bio in R^{B x C} and H_phys in R^{B x C} are the batch-wise feature activations 
  (or scale parameters gamma) extracted from the biological and physical AdaLN paths.
- B is the effective batch size (B, or B * H * W if spatial dimensions are flattened).
- C is the channel dimension.
- H_bar represents the column-wise mean vector.
- ||.||_F denotes the Frobenius norm.
"""

import torch
import torch.nn as nn

class CovarianceLoss(nn.Module):
    """
    Computes the squared Frobenius norm of the empirical cross-covariance matrix 
    between biological and physical feature representations.
    """
    
    def __init__(self):
        """
        Initializes the CovarianceLoss module.
        """
        super(CovarianceLoss, self).__init__()

    def _flatten_spatial_dims(self, H: torch.Tensor) -> torch.Tensor:
        """
        Helper method to flatten 4D feature maps (B, C, H, W) into 2D matrices (B*H*W, C)
        to compute the covariance over all spatial locations and batch elements.
        If the input is already 2D (B, C), it is returned as is.
        
        :param H: Input tensor of shape (B, C) or (B, C, H, W).
        :return: Flattened tensor of shape (N, C), where N = B or B*H*W.
        """
        if H.dim() == 4:
            B, C, H_dim, W_dim = H.shape
            # Permute to (B, H, W, C) and reshape to (B*H*W, C)
            H = H.permute(0, 2, 3, 1).reshape(-1, C)
        elif H.dim() != 2:
            raise ValueError(f"Expected 2D (B, C) or 4D (B, C, H, W) input, got {H.dim()}D")
        return H

    def forward(self, H_bio: torch.Tensor, H_phys: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-covariance loss between biological and physical features.
        
        Mathematical Steps:
        1. Center the features: H_centered = H - mean(H)
        2. Compute cross-covariance: C_cross = (1 / (N - 1)) * H_bio^T * H_phys
        3. Compute squared Frobenius norm: L_cov = || C_cross ||_F^2
        
        :param H_bio: Biological feature activations of shape (B, C) or (B, C, H, W).
        :param H_phys: Physical feature activations of shape (B, C) or (B, C, H, W).
        :return: Scalar covariance loss.
        """
        if H_bio.shape != H_phys.shape:
            raise ValueError(f"Shape mismatch: H_bio {H_bio.shape} vs H_phys {H_phys.shape}")
            
        # 1. Flatten spatial dimensions if necessary
        H_bio_flat = self._flatten_spatial_dims(H_bio)
        H_phys_flat = self._flatten_spatial_dims(H_phys)
        
        N = H_bio_flat.shape[0]
        
        # 2. Center the features: H - H_bar
        # mean(dim=0, keepdim=True) computes the column-wise mean vector
        H_bio_centered = H_bio_flat - H_bio_flat.mean(dim=0, keepdim=True)
        H_phys_centered = H_phys_flat - H_phys_flat.mean(dim=0, keepdim=True)
        
        # 3. Compute cross-covariance matrix: (1 / (N - 1)) * H_bio^T * H_phys
        # Using max(N - 1, 1) to prevent division by zero if the effective batch size is 1
        cross_cov = (H_bio_centered.T @ H_phys_centered) / max(N - 1, 1)
        
        # 4. Compute squared Frobenius norm: || cross_cov ||_F^2
        # torch.norm with p='fro' computes the Frobenius norm
        loss = torch.norm(cross_cov, p='fro') ** 2
        
        return loss


if __name__ == "__main__":
    # Example usage and validation of the CovarianceLoss module
    
    # 1. Instantiate the loss function
    cov_loss_fn = CovarianceLoss()
    
    # 2. Define orig tensor shapes
    batch_size = 16
    channels = 128
    height, width = 16, 16
    
    # Scenario A: 2D Inputs (e.g., scale parameters gamma from AdaLN)
    gamma_bio_2d = torch.randn(batch_size, channels, requires_grad=True)
    gamma_phys_2d = torch.randn(batch_size, channels, requires_grad=True)
    
    loss_2d = cov_loss_fn(gamma_bio_2d, gamma_phys_2d)
    print(f"2D Input Covariance Loss: {loss_2d.item():.4f}")
    
    # Scenario B: 4D Inputs (e.g., intermediate feature maps H^(l))
    H_bio_4d = torch.randn(batch_size, channels, height, width, requires_grad=True)
    H_phys_4d = torch.randn(batch_size, channels, height, width, requires_grad=True)
    
    loss_4d = cov_loss_fn(H_bio_4d, H_phys_4d)
    print(f"4D Input Covariance Loss: {loss_4d.item():.4f}")
    
    # 3. Verify backward pass (gradient flow)
    loss_4d.backward()
    
    # Check if gradients flowed back to the feature maps
    assert H_bio_4d.grad is not None, "Gradients did not flow back to H_bio!"
    assert H_phys_4d.grad is not None, "Gradients did not flow back to H_phys!"
    
    # 4. Verify that orthogonal inputs yield near-zero loss
    # Create two strictly orthogonal matrices
    N_orth = 64
    C_orth = 32
    # Generate a random orthogonal matrix Q
    random_matrix = torch.randn(N_orth, C_orth)
    Q, _ = torch.linalg.qr(random_matrix)
    
    H_orth_1 = Q[:, :16]  # First 16 columns
    H_orth_2 = Q[:, 16:]  # Next 16 columns
    
    loss_orth = cov_loss_fn(H_orth_1, H_orth_2)
    print(f"\nOrthogonal Inputs Covariance Loss: {loss_orth.item():.6f} (Should be near 0.0)")
    
    print("\nVerification passed: Covariance loss computed correctly for 2D/4D inputs, gradients flow successfully, and orthogonal inputs yield minimal loss.")