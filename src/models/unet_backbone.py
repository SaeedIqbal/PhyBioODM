"""
unet_backbone.py
Implements the Base UNet / Diffusion UNet architecture for the PhyBio-ODM framework.
This module provides the core neural network backbone that processes the noisy image 
x_t and timestep t, integrating the orthogonal dual-stream conditioning via AdaLN.

Mathematical Formulations:
1. Sinusoidal Time Embedding:
   PE(t, 2i) = sin(t / 10000^{2i/d})
   PE(t, 2i+1) = cos(t / 10000^{2i/d})

2. Residual Block with Orthogonal AdaLN Conditioning:
   h = GroupNorm(x)
   h, gamma_bio, gamma_phys = OrthogonalAdaLN(h, v_bio, v_phys)
   h = Swish(h + MLP(time_emb))
   h = Conv(h)
   ... (second conv layer)
   x_out = x + h
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Import the OrthogonalAdaLN module defined in the conditioning package
# This reuses the dual-stream sequential modulation logic to resolve Gap 2.
from .orthogonal_adaln import OrthogonalAdaLN


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Computes sinusoidal positional embeddings for the diffusion timestep t.
    This allows the network to understand the noise level (time) in the diffusion process.
    
    Mathematical Equation:
    PE(t, 2i) = sin(t / 10000^{2i/d_model})
    PE(t, 2i+1) = cos(t / 10000^{2i/d_model})
    """
    
    def __init__(self, dim: int):
        """
        :param dim: Dimension of the embedding space (d_model).
        """
        super(SinusoidalPositionalEmbedding, self).__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("Dimension must be a strictly positive even integer.")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        :param t: Timestep tensor of shape (batch_size,).
        :return: Sinusoidal embedding of shape (batch_size, dim).
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Compute the denominator: 10000^{2i/d_model}
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # Multiply by timestep t: t / 10000^{2i/d_model}
        emb = t[:, None] * emb[None, :]
        
        # Apply sin and cos to even and odd indices respectively
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """
    Residual block incorporating time embedding and orthogonal dual-stream AdaLN conditioning.
    Reuses the OrthogonalAdaLN module to enforce strict biological-physical disentanglement.
    
    Mathematical Formulation:
    1. h = GroupNorm(x)
    2. h, gamma_bio, gamma_phys = OrthogonalAdaLN(h, v_bio, v_phys)
    3. h = Swish(h + TimeEmb(t))
    4. h = Conv(h)
    5. h = GroupNorm(h)
    6. h = Swish(h)
    7. h = Dropout(h)
    8. h = Conv(h)
    9. x_out = x + h (with channel projection if in_channels != out_channels)
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 bio_emb_dim: int, phys_emb_dim: int, dropout: float = 0.0):
        """
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param time_emb_dim: Dimension of the time embedding.
        :param bio_emb_dim: Dimension of the biological latent embedding (d_bio).
        :param phys_emb_dim: Dimension of the physical latent embedding (d_phys).
        :param dropout: Dropout probability for regularization.
        """
        super(ResidualBlock, self).__init__()
        
        # First normalization and orthogonal AdaLN injection
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        self.adaLN = OrthogonalAdaLN(
            feature_dim=in_channels, 
            bio_embed_dim=bio_emb_dim, 
            phys_embed_dim=phys_emb_dim
        )
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, in_channels)
        )
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Second normalization and convolution
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Residual projection if channel dimensions change
        self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, v_bio: torch.Tensor, v_phys: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input feature map of shape (B, C_in, H, W).
        :param t_emb: Time embedding of shape (B, time_emb_dim).
        :param v_bio: Biological latent embedding of shape (B, d_bio).
        :param v_phys: Physical latent embedding of shape (B, d_phys).
        :return: Output feature map of shape (B, C_out, H, W).
        """
        h = self.norm1(x)
        
        # Apply Orthogonal AdaLN (reusing the dual-stream modulation)
        # We only need the modulated feature map 'h' here; the gammas are used for the loss externally.
        h, _, _ = self.adaLN(h, v_bio, v_phys)
        
        # Add time embedding
        t_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_proj
        h = F.silu(h)
        
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Add residual connection
        return h + self.residual_proj(x)


class Downsample(nn.Module):
    """
    Spatial downsampling module using strided convolution.
    """
    def __init__(self, channels: int):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Spatial upsampling module using nearest-neighbor interpolation followed by convolution.
    """
    def __init__(self, channels: int):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNetBackbone(nn.Module):
    """
    The core UNet architecture for the PhyBio-ODM diffusion model.
    Processes the noisy image x_t and timestep t, conditioned on v_bio and v_phys.
    """
    
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3, 
                 base_channels: int = 64, 
                 channel_multipliers: List[int] = [1, 2, 4], 
                 time_emb_dim: int = 256,
                 bio_emb_dim: int = 512,
                 phys_emb_dim: int = 256,
                 dropout: float = 0.1):
        """
        :param in_channels: Number of input image channels (e.g., 3 for RGB).
        :param out_channels: Number of output channels (e.g., 3 for noise prediction).
        :param base_channels: Base number of channels in the UNet.
        :param channel_multipliers: Channel multipliers for each resolution level.
        :param time_emb_dim: Dimension of the time embedding.
        :param bio_emb_dim: Dimension of the biological latent embedding.
        :param phys_emb_dim: Dimension of the physical latent embedding.
        :param dropout: Dropout rate for ResidualBlocks.
        """
        super(UNetBackbone, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        
        # 1. Time Embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # 2. Initial Convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 3. Downsampling Path (Encoder)
        self.downs = nn.ModuleList()
        channels = [base_channels]
        current_channels = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult
            
            # Add two residual blocks per resolution level
            self.downs.append(nn.ModuleList([
                ResidualBlock(current_channels, out_channels, time_emb_dim, bio_emb_dim, phys_emb_dim, dropout),
                ResidualBlock(out_channels, out_channels, time_emb_dim, bio_emb_dim, phys_emb_dim, dropout)
            ]))
            channels.extend([out_channels, out_channels])
            current_channels = out_channels
            
            # Add downsample if not the last level
            if i != len(channel_multipliers) - 1:
                self.downs.append(nn.ModuleList([Downsample(current_channels), None]))
                channels.append(current_channels)
                
        # 4. Middle Block
        mid_channels = base_channels * channel_multipliers[-1]
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels, time_emb_dim, bio_emb_dim, phys_emb_dim, dropout)
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels, time_emb_dim, bio_emb_dim, phys_emb_dim, dropout)
        
        # 5. Upsampling Path (Decoder)
        self.ups = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_channels = base_channels * mult
            
            # Skip connection concatenation doubles the input channels
            in_ch = current_channels + channels.pop()
            
            self.ups.append(nn.ModuleList([
                ResidualBlock(in_ch, out_channels, time_emb_dim, bio_emb_dim, phys_emb_dim, dropout),
                ResidualBlock(out_channels, out_channels, time_emb_dim, bio_emb_dim, phys_emb_dim, dropout)
            ]))
            current_channels = out_channels
            channels.pop() # Pop the second block's channel count
            channels.pop() # Pop the first block's channel count
            
            # Add upsample if not the first level
            if i != 0:
                self.ups.append(nn.ModuleList([Upsample(current_channels), None]))
                
        # 6. Final Output Convolution
        self.final_norm = nn.GroupNorm(num_groups=min(32, current_channels), num_channels=current_channels)
        self.final_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the final convolution layer with zeros to ensure that 
        the initial noise prediction is close to zero, stabilizing early training.
        """
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, v_bio: torch.Tensor, v_phys: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet backbone.
        
        :param x: Noisy input image of shape (B, C_in, H, W).
        :param t: Diffusion timestep of shape (B,).
        :param v_bio: Biological latent embedding of shape (B, d_bio).
        :param v_phys: Physical latent embedding of shape (B, d_phys).
        :return: Predicted noise or clean image of shape (B, C_out, H, W).
        """
        # 1. Compute time embedding
        t_emb = self.time_mlp(t.float())
        
        # 2. Initial convolution
        h = self.init_conv(x)
        h_stack = [h]
        
        # 3. Downsampling path
        for block_group in self.downs:
            block1, block2 = block_group[0], block_group[1]
            
            if block2 is None: 
                # This is a Downsample module
                h = block1(h)
            else:
                # These are ResidualBlocks
                h = block1(h, t_emb, v_bio, v_phys)
                h_stack.append(h)
                h = block2(h, t_emb, v_bio, v_phys)
                h_stack.append(h)
                
        # 4. Middle blocks
        h = self.mid_block1(h, t_emb, v_bio, v_phys)
        h = self.mid_block2(h, t_emb, v_bio, v_phys)
        
        # 5. Upsampling path with skip connections
        for block_group in self.ups:
            block1, block2 = block_group[0], block_group[1]
            
            if block2 is None:
                # This is an Upsample module
                h = block1(h)
            else:
                # Concatenate skip connection
                h = torch.cat([h, h_stack.pop()], dim=1)
                h = block1(h, t_emb, v_bio, v_phys)
                h = torch.cat([h, h_stack.pop()], dim=1)
                h = block2(h, t_emb, v_bio, v_phys)
                
        # 6. Final output
        h = self.final_norm(h)
        h = F.silu(h)
        out = self.final_conv(h)
        
        return out


if __name__ == "__main__":
    # Example usage and validation of the UNetBackbone
    
    # 1. Define hyperparameters
    batch_size = 4
    in_channels = 3
    out_channels = 3
    height, width = 64, 64
    base_channels = 32
    channel_multipliers = [1, 2, 4]
    time_emb_dim = 128
    bio_emb_dim = 256
    phys_emb_dim = 128
    
    # 2. Instantiate the UNet Backbone
    unet = UNetBackbone(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        time_emb_dim=time_emb_dim,
        bio_emb_dim=bio_emb_dim,
        phys_emb_dim=phys_emb_dim,
        dropout=0.1
    )
    
    print(f"Initialized UNetBackbone:")
    print(f"  Input Channels: {in_channels}")
    print(f"  Base Channels: {base_channels}")
    print(f"  Channel Multipliers: {channel_multipliers}")
    print(f"  Total Parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # 3. Create orig inputs
    x_orig = torch.randn(batch_size, in_channels, height, width)
    t_orig = torch.randint(0, 1000, (batch_size,))
    v_bio_orig = torch.randn(batch_size, bio_emb_dim)
    v_phys_orig = torch.randn(batch_size, phys_emb_dim)
    
    # 4. Forward pass
    out = unet(x_orig, t_orig, v_bio_orig, v_phys_orig)
    
    print(f"\nForward Pass Validation:")
    print(f"  Input x shape: {x_orig.shape}")
    print(f"  Output shape: {out.shape}")
    
    # 5. Verify output shape
    assert out.shape == (batch_size, out_channels, height, width), "Output shape mismatch!"
    
    # 6. Test backward pass
    loss = out.sum()
    loss.backward()
    
    # Check if gradients flowed to the initial conv and final conv
    assert unet.init_conv.weight.grad is not None, "Gradients did not flow to init_conv!"
    assert unet.final_conv.weight.grad is not None, "Gradients did not flow to final_conv!"
    
    print("\nVerification passed: UNetBackbone shapes are correct and gradients flow successfully.")