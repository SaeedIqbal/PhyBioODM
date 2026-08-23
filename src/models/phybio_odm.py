"""
phybio_odm.py
Implements the full proposed PhyBio-ODM (Physics-Biology Orthogonal Diffusion Model).
This module integrates the continuous physical parameter projection, 
biology embedding, and the orthogonal dual-stream UNet backbone.

Mathematical Formulations:
1. Biological Embedding: v_bio = E_class(y)
2. Physical Embedding: v_phys = phi_phys(p)
3. Noise Prediction: epsilon_pred = UNet(x_t, t, v_bio, v_phys)
"""

import torch
import torch.nn as nn
from typing import List

# Relative imports for the proposed methodology components
from .unet_backbone import UNetBackbone
from ..conditioning.continuous_mlp import PhysicsAwareMLP

class PhyBioODM(nn.Module):
    """
    Full PhyBio-ODM model integrating orthogonal dual-stream conditioning 
    and continuous physical parameter embeddings.
    """
    
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 base_channels: int = 64,
                 channel_multipliers: List[int] = [1, 2, 4],
                 time_emb_dim: int = 256,
                 bio_emb_dim: int = 512,
                 phys_param_dim: int = 10,
                 phys_hidden_dims: List[int] = [64, 128],
                 phys_emb_dim: int = 256,
                 dropout: float = 0.1):
        """
        Initializes the PhyBio-ODM model.
        
        :param num_classes: Number of discrete biological classes (K).
        :param in_channels: Number of input image channels (e.g., 3 for RGB).
        :param out_channels: Number of output channels (e.g., 3 for noise prediction).
        :param base_channels: Base number of channels in the UNet.
        :param channel_multipliers: Channel multipliers for each resolution level.
        :param time_emb_dim: Dimension of the time embedding.
        :param bio_emb_dim: Dimension of the biological latent embedding (d_bio).
        :param phys_param_dim: Dimension of the continuous physical parameter vector (d).
        :param phys_hidden_dims: Hidden layer dimensions for the physics-aware MLP.
        :param phys_emb_dim: Dimension of the physical latent embedding (d_phys).
        :param dropout: Dropout rate for regularization.
        """
        super(PhyBioODM, self).__init__()
        
        if num_classes <= 0:
            raise ValueError("num_classes must be strictly positive.")
        if phys_param_dim <= 0:
            raise ValueError("phys_param_dim must be strictly positive.")
            
        self.num_classes = num_classes
        self.bio_emb_dim = bio_emb_dim
        self.phys_emb_dim = phys_emb_dim
        
        # 1. Biological Embedding: E_class(y)
        # Maps discrete class label y to continuous biological latent space v_bio
        self.E_class = nn.Embedding(num_embeddings=num_classes, embedding_dim=bio_emb_dim)
        
        # 2. Physical Embedding: phi_phys(p)
        # Maps continuous physical parameter vector p to continuous physical latent space v_phys
        self.phi_phys = PhysicsAwareMLP(
            input_dim=phys_param_dim,
            hidden_dims=phys_hidden_dims,
            output_dim=phys_emb_dim,
            dropout_rate=dropout,
            use_batch_norm=True
        )
        
        # 3. UNet Backbone
        # Processes noisy image x_t conditioned on v_bio and v_phys via Orthogonal AdaLN
        self.unet = UNetBackbone(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            time_emb_dim=time_emb_dim,
            bio_emb_dim=bio_emb_dim,
            phys_emb_dim=phys_emb_dim,
            dropout=dropout
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the biological embedding matrix using a normal distribution 
        to ensure stable gradient flow during the training of the orthogonal dual-stream architecture.
        """
        nn.init.normal_(self.E_class.weight, mean=0.0, std=0.02)

    def get_biological_embedding(self, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the biological latent embedding.
        
        Equation: v_bio = E_class(y)
        
        :param y: Tensor of discrete class labels of shape (batch_size,).
        :return: Biological latent embedding v_bio of shape (batch_size, d_bio).
        """
        if y.dim() != 1:
            raise ValueError(f"Expected class labels y to be 1D (batch_size,), got {y.dim()}D.")
        return self.E_class(y)

    def get_physical_embedding(self, p: torch.Tensor) -> torch.Tensor:
        """
        Computes the physical latent embedding.
        
        Equation: v_phys = phi_phys(p)
        
        :param p: Continuous physical parameter vector of shape (batch_size, d).
        :return: Physical latent embedding v_phys of shape (batch_size, d_phys).
        """
        return self.phi_phys(p)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PhyBio-ODM model.
        
        Mathematical Formulation:
        epsilon_pred = UNet(x_t, t, E_class(y), phi_phys(p))
        
        :param x_t: Noisy input image of shape (B, C_in, H, W).
        :param t: Diffusion timestep of shape (B,).
        :param y: Discrete class label of shape (B,).
        :param p: Continuous physical parameter vector of shape (B, d).
        :return: Predicted noise epsilon_pred of shape (B, C_out, H, W).
        """
        if x_t.dim() != 4:
            raise ValueError(f"Expected input image x_t to be 4D (B, C, H, W), got {x_t.dim()}D.")
        if t.dim() != 1:
            raise ValueError(f"Expected timestep t to be 1D (B,), got {t.dim()}D.")
            
        # 1. Compute Biological Embedding: v_bio = E_class(y)
        v_bio = self.get_biological_embedding(y)
        
        # 2. Compute Physical Embedding: v_phys = phi_phys(p)
        v_phys = self.get_physical_embedding(p)
        
        # 3. Predict Noise via UNet Backbone
        epsilon_pred = self.unet(x_t, t, v_bio, v_phys)
        
        return epsilon_pred


if __name__ == "__main__":
    # Example usage and validation of the full PhyBio-ODM model
    
    # 1. Define hyperparameters
    batch_size = 4
    num_classes = 32
    in_channels = 3
    out_channels = 3
    height, width = 64, 64
    base_channels = 32
    channel_multipliers = [1, 2, 4]
    time_emb_dim = 128
    bio_emb_dim = 256
    phys_param_dim = 10  # Matches PhysicalParameterAggregator output
    phys_hidden_dims = [64, 128]
    phys_emb_dim = 128
    
    # 2. Instantiate the PhyBio-ODM model
    model = PhyBioODM(
        num_classes=num_classes,
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        time_emb_dim=time_emb_dim,
        bio_emb_dim=bio_emb_dim,
        phys_param_dim=phys_param_dim,
        phys_hidden_dims=phys_hidden_dims,
        phys_emb_dim=phys_emb_dim,
        dropout=0.1
    )
    
    print(f"Initialized PhyBio-ODM:")
    print(f"  Number of Classes (K): {num_classes}")
    print(f"  Biological Embedding Dim (d_bio): {bio_emb_dim}")
    print(f"  Physical Parameter Dim (d): {phys_param_dim}")
    print(f"  Physical Embedding Dim (d_phys): {phys_emb_dim}")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Create orig inputs
    x_t_orig = torch.randn(batch_size, in_channels, height, width)
    t_orig = torch.randint(0, 1000, (batch_size,))
    y_orig = torch.randint(0, num_classes, (batch_size,))
    p_orig = torch.randn(batch_size, phys_param_dim)
    
    # 4. Forward pass
    epsilon_pred = model(x_t_orig, t_orig, y_orig, p_orig)
    
    print(f"\nForward Pass Validation:")
    print(f"  Input x_t shape: {x_t_orig.shape}")
    print(f"  Output epsilon_pred shape: {epsilon_pred.shape}")
    
    # 5. Verify output shape
    assert epsilon_pred.shape == (batch_size, out_channels, height, width), "Output shape mismatch!"
    
    # 6. Test backward pass
    loss = epsilon_pred.sum()
    loss.backward()
    
    # Check if gradients flowed to the embeddings and UNet
    assert model.E_class.weight.grad is not None, "Gradients did not flow to E_class!"
    assert model.phi_phys.phi_phys[0].weight.grad is not None, "Gradients did not flow to phi_phys!"
    assert model.unet.init_conv.weight.grad is not None, "Gradients did not flow to UNet!"
    
    print("\nVerification passed: PhyBio-ODM shapes are correct and gradients flow successfully through all components.")