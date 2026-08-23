"""
diffusion_loss.py
Implements the standard diffusion loss (noise prediction Mean Squared Error) 
for the PhyBio-ODM framework. This loss drives the reverse diffusion process 
by minimizing the difference between the predicted noise and the actual noise 
added during the forward process.

Mathematical Formulation:
L_diff = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(x_t, t, c) ||_2^2 ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DiffusionLoss(nn.Module):
    """
    Computes the standard diffusion loss (Mean Squared Error) between 
    the predicted noise and the true noise added during the forward diffusion process.
    
    This is the primary objective for training the score network (UNet) in 
    the PhyBio-ODM framework, ensuring accurate noise prediction which 
    translates to high-fidelity image synthesis.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initializes the DiffusionLoss module.
        
        :param reduction: Specifies the reduction to apply to the output: 
                          'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super(DiffusionLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("reduction must be one of 'none', 'mean', or 'sum'.")
        self.reduction = reduction

    def forward(self, epsilon_pred: torch.Tensor, epsilon_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the Mean Squared Error (MSE) between predicted and true noise.
        
        Mathematical Equation:
        L_diff = || epsilon_true - epsilon_pred ||_2^2
        
        :param epsilon_pred: Predicted noise tensor of shape (B, C, H, W).
        :param epsilon_true: True noise tensor of shape (B, C, H, W).
        :return: Scalar diffusion loss (if reduction is 'mean' or 'sum') 
                 or element-wise loss tensor (if reduction is 'none').
        """
        if epsilon_pred.shape != epsilon_true.shape:
            raise ValueError(f"Shape mismatch: epsilon_pred {epsilon_pred.shape} vs epsilon_true {epsilon_true.shape}")
            
        # Compute the squared L2 norm (MSE)
        # || epsilon_true - epsilon_pred ||_2^2
        loss = F.mse_loss(epsilon_pred, epsilon_true, reduction=self.reduction)
        
        return loss


if __name__ == "__main__":
    # Example usage and validation of the DiffusionLoss module
    
    # 1. Instantiate the loss function
    loss_fn = DiffusionLoss(reduction='mean')
    
    # 2. Define orig tensor shapes
    batch_size = 4
    channels = 3
    height, width = 64, 64
    
    # orig predicted and true noise (simulating the output of the UNet and the forward process)
    eps_pred = torch.randn(batch_size, channels, height, width, requires_grad=True)
    eps_true = torch.randn(batch_size, channels, height, width)
    
    # 3. Compute the diffusion loss
    loss = loss_fn(eps_pred, eps_true)
    
    print(f"Initialized DiffusionLoss:")
    print(f"  Reduction method: {loss_fn.reduction}")
    print(f"\nForward Pass Validation:")
    print(f"  Predicted noise (epsilon_pred) shape: {eps_pred.shape}")
    print(f"  True noise (epsilon_true) shape: {eps_true.shape}")
    print(f"  Diffusion Loss (L_diff): {loss.item():.4f}")
    
    # 4. Verify backward pass (gradient flow)
    loss.backward()
    
    # Check if gradients flowed back to the predicted noise tensor
    assert eps_pred.grad is not None, "Gradients did not flow back to epsilon_pred!"
    
    print("\nVerification passed: Diffusion loss computed correctly and gradients flow successfully.")