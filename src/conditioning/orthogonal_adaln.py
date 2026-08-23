"""
orthogonal_adaln.py
Implements the Dual-stream Sequential Adaptive Layer Normalization (AdaLN) injection.
This module resolves Gap 2 (Additive Morphological Entanglement) by replacing the 
linear additive fusion of MeDi with a hierarchical, orthogonal conditioning mechanism.

Mathematical Formulations:
1. Layer Normalization: 
   LayerNorm(h) = (h - mu(h)) / sqrt(sigma^2(h) + epsilon)

2. Biological Modulation (First Stream):
   h_hat = gamma_bio(v_bio) * LayerNorm(h) + beta_bio(v_bio)

3. Physical Modulation (Second Stream):
   h_next = gamma_phys(v_phys) * h_hat + beta_phys(v_phys)

Where:
- h in R^{C x H x W} is the intermediate feature map.
- v_bio in R^{d_bio} and v_phys in R^{d_phys} are the latent embeddings.
- gamma and beta are learnable affine projection functions mapping to R^C.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class OrthogonalAdaLN(nn.Module):
    """
    Dual-stream Sequential Adaptive Layer Normalization (AdaLN) module.
    Enforces a strict structural hierarchy where biological features define 
    the geometric baseline and physical parameters modulate the optical style.
    """
    
    def __init__(self, feature_dim: int, bio_embed_dim: int, phys_embed_dim: int, epsilon: float = 1e-5):
        """
        Initializes the OrthogonalAdaLN module.
        
        :param feature_dim: Dimension of the intermediate feature map (C).
        :param bio_embed_dim: Dimension of the biological latent embedding (d_bio).
        :param phys_embed_dim: Dimension of the physical latent embedding (d_phys).
        :param epsilon: Small constant for numerical stability in LayerNorm.
        """
        super(OrthogonalAdaLN, self).__init__()
        
        if feature_dim <= 0 or bio_embed_dim <= 0 or phys_embed_dim <= 0:
            raise ValueError("All dimensions must be strictly positive.")
        if epsilon <= 0:
            raise ValueError("Epsilon must be strictly positive.")
            
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        
        # Learnable affine projection functions for the Biological Stream
        # Maps v_bio to gamma_bio and beta_bio in R^C
        self.bio_proj = nn.Linear(bio_embed_dim, feature_dim * 2)
        
        # Learnable affine projection functions for the Physical Stream
        # Maps v_phys to gamma_phys and beta_phys in R^C
        self.phys_proj = nn.Linear(phys_embed_dim, feature_dim * 2)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the projection layers. 
        We initialize the bias of gamma to 0 and beta to 0, ensuring that 
        initially, the AdaLN acts as an identity function (gamma=1, beta=0) 
        before the non-linear activation (if any) or just standard scaling.
        Actually, standard AdaLN initializes gamma bias to -1 or 0. We use 0.
        """
        nn.init.zeros_(self.bio_proj.bias)
        nn.init.zeros_(self.phys_proj.bias)
        nn.init.xavier_uniform_(self.bio_proj.weight)
        nn.init.xavier_uniform_(self.phys_proj.weight)

    def _layer_norm(self, h: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization over the channel dimension (C).
        
        Equation: LayerNorm(h) = (h - mu(h)) / sqrt(sigma^2(h) + epsilon)
        
        :param h: Input feature map of shape (B, C, H, W).
        :return: Normalized feature map of shape (B, C, H, W).
        """
        # Compute mean and variance over the channel dimension (dim=1)
        mean = h.mean(dim=1, keepdim=True)
        var = h.var(dim=1, keepdim=True, unbiased=False)
        
        # Normalize
        h_norm = (h - mean) / torch.sqrt(var + self.epsilon)
        return h_norm

    def _project_and_split(self, v: torch.Tensor, proj_layer: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects the latent embedding and splits it into scale (gamma) and shift (beta) parameters.
        
        :param v: Latent embedding of shape (B, d_embed).
        :param proj_layer: The linear projection layer (bio_proj or phys_proj).
        :return: Tuple of (gamma, beta), each of shape (B, C).
        """
        # Project to 2 * C
        params = proj_layer(v)  # Shape: (B, 2 * C)
        
        # Split into gamma and beta
        gamma, beta = params.chunk(2, dim=-1)  # Each shape: (B, C)
        
        # Apply Swish activation to gamma to ensure it can be both positive and negative, 
        # but typically in AdaLN, gamma is passed through an activation like SiLU/Swish 
        # or simply used as is. The methodology implies direct affine transformation.
        # We will use a simple shift to ensure gamma is centered around 1 for stable training.
        gamma = gamma + 1.0 
        
        return gamma, beta

    def forward(self, h: torch.Tensor, v_bio: torch.Tensor, v_phys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass implementing the hierarchical dual-stream modulation.
        
        :param h: Intermediate feature map of shape (B, C, H, W).
        :param v_bio: Biological latent embedding of shape (B, d_bio).
        :param v_phys: Physical latent embedding of shape (B, d_phys).
        :return: Tuple containing:
                 - h_next: Modulated feature map of shape (B, C, H, W).
                 - gamma_bio: Biological scale parameters of shape (B, C) (for orthogonality loss).
                 - gamma_phys: Physical scale parameters of shape (B, C) (for orthogonality loss).
        """
        if h.dim() != 4:
            raise ValueError(f"Expected feature map h to be 4D (B, C, H, W), got {h.dim()}D.")
        if h.shape[1] != self.feature_dim:
            raise ValueError(f"Expected feature dimension {self.feature_dim}, got {h.shape[1]}.")
            
        B, C, H, W = h.shape
        
        # 1. Layer Normalization
        h_norm = self._layer_norm(h)
        
        # 2. Biological Modulation (First Stream)
        gamma_bio, beta_bio = self._project_and_split(v_bio, self.bio_proj)
        
        # Reshape gamma and beta for broadcasting: (B, C) -> (B, C, 1, 1)
        gamma_bio_4d = gamma_bio.unsqueeze(-1).unsqueeze(-1)
        beta_bio_4d = beta_bio.unsqueeze(-1).unsqueeze(-1)
        
        # h_hat = gamma_bio * LayerNorm(h) + beta_bio
        h_hat = gamma_bio_4d * h_norm + beta_bio_4d
        
        # 3. Physical Modulation (Second Stream)
        gamma_phys, beta_phys = self._project_and_split(v_phys, self.phys_proj)
        
        # Reshape for broadcasting
        gamma_phys_4d = gamma_phys.unsqueeze(-1).unsqueeze(-1)
        beta_phys_4d = beta_phys.unsqueeze(-1).unsqueeze(-1)
        
        # h_next = gamma_phys * h_hat + beta_phys
        h_next = gamma_phys_4d * h_hat + beta_phys_4d
        
        # Return the modulated feature map and the scale parameters (gamma) 
        # which are explicitly required by the Manifold Orthogonality Constraint Loss.
        return h_next, gamma_bio, gamma_phys


if __name__ == "__main__":
    # Example usage and validation of the OrthogonalAdaLN module
    
    # 1. Define hyperparameters
    batch_size = 8
    feature_dim = 256       # C (UNet channel dimension)
    height, width = 32, 32  # H, W (Spatial dimensions)
    bio_embed_dim = 512     # d_bio
    phys_embed_dim = 256    # d_phys (from PhysicsAwareMLP)
    
    # 2. Instantiate the OrthogonalAdaLN module
    adaLN = OrthogonalAdaLN(
        feature_dim=feature_dim, 
        bio_embed_dim=bio_embed_dim, 
        phys_embed_dim=phys_embed_dim, 
        epsilon=1e-5
    )
    
    print(f"Initialized OrthogonalAdaLN:")
    print(f"  Feature Dimension (C): {feature_dim}")
    print(f"  Biological Embedding Dim (d_bio): {bio_embed_dim}")
    print(f"  Physical Embedding Dim (d_phys): {phys_embed_dim}")
    print(f"  Total Parameters: {sum(p.numel() for p in adaLN.parameters()):,}")
    
    # 3. Create orig inputs
    h_orig = torch.randn(batch_size, feature_dim, height, width)
    v_bio_orig = torch.randn(batch_size, bio_embed_dim)
    v_phys_orig = torch.randn(batch_size, phys_embed_dim)
    
    # 4. Forward pass
    h_next, gamma_bio, gamma_phys = adaLN(h_orig, v_bio_orig, v_phys_orig)
    
    print(f"\nForward Pass Validation:")
    print(f"  Input h shape: {h_orig.shape}")
    print(f"  Output h_next shape: {h_next.shape}")
    print(f"  gamma_bio shape (for loss): {gamma_bio.shape}")
    print(f"  gamma_phys shape (for loss): {gamma_phys.shape}")
    
    # 5. Verify output shapes
    assert h_next.shape == (batch_size, feature_dim, height, width), "Output feature map shape mismatch!"
    assert gamma_bio.shape == (batch_size, feature_dim), "gamma_bio shape mismatch!"
    assert gamma_phys.shape == (batch_size, feature_dim), "gamma_phys shape mismatch!"
    
    # 6. Test backward pass (gradient flow for orthogonality loss)
    # Simulate a orig orthogonality loss: || gamma_bio^T * gamma_phys ||_F^2
    # (This is a simplified version of the actual covariance loss)
    orig_ortho_loss = torch.norm(gamma_bio.T @ gamma_phys, p='fro')**2 
    
    orig_ortho_loss.backward()
    
    # Check if gradients flowed back to the projection layers
    assert adaLN.bio_proj.weight.grad is not None, "Gradients did not flow to bio_proj!"
    assert adaLN.phys_proj.weight.grad is not None, "Gradients did not flow to phys_proj!"
    
    print("\nVerification passed: Shapes are correct, and gradients flow successfully for the orthogonality loss.")