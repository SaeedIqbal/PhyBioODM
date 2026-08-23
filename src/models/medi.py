"""
medi.py
Implements the MeDi (Metadata-guided Diffusion) baseline model.
This module represents the state-of-the-art baseline that relies on discrete token 
embeddings and additive conditioning. This approach causes the "Additive Morphological 
Entanglement" (Gap 2) that PhyBio-ODM aims to resolve by forcing orthogonal biological 
and physical factors into a shared Euclidean subspace.

Mathematical Formulations:
1. Class Embedding: E_class(y)
2. Metadata Embeddings: sum_{j=1}^{J} E_meta^{(j)}(m_j)
3. Additive Conditioning: z_cond = E_class(y) + sum_{j=1}^{J} E_meta^{(j)}(m_j)
4. Time Embedding: z_t = MLP(t)
5. Final Conditioning (Additive Fusion): z_final = z_t + z_cond
6. Noise Prediction: epsilon_pred = UNet(x_t, z_final)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

# Reuse components from the proposed methodology and conditioning packages
from .unet_backbone import SinusoidalPositionalEmbedding
from ..conditioning.discrete_embedding import DiscreteEmbedding


class MeDiResidualBlock(nn.Module):
    """
    Residual block for the MeDi baseline.
    Unlike PhyBio-ODM's dual-stream AdaLN, this block uses standard FiLM/AdaLN 
    with a single, additively fused conditioning vector z_final.
    
    Mathematical Formulation:
    1. h = GroupNorm(x)
    2. gamma, beta = Linear(z_final)
    3. h = (gamma + 1) * h + beta
    4. h = SiLU(h)
    5. h = Conv(h) ... (second conv layer)
    6. x_out = x + h
    """
    
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.0):
        """
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param cond_dim: Dimension of the fused conditioning vector (z_final).
        :param dropout: Dropout probability for regularization.
        """
        super(MeDiResidualBlock, self).__init__()
        
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        
        # Single projection layer for the additively fused conditioning vector
        self.cond_proj = nn.Linear(cond_dim, in_channels * 2)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Residual projection if channel dimensions change
        self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, z_final: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input feature map of shape (B, C_in, H, W).
        :param z_final: Fused conditioning vector of shape (B, cond_dim).
        :return: Output feature map of shape (B, C_out, H, W).
        """
        h = self.norm1(x)
        
        # Standard AdaLN / FiLM with single conditioning vector
        params = self.cond_proj(z_final)
        gamma, beta = params.chunk(2, dim=-1)
        
        # Reshape for broadcasting: (B, C) -> (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) + 1.0
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        h = gamma * h + beta
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Add residual connection
        return h + self.residual_proj(x)


class Downsample(nn.Module):
    """Spatial downsampling module using strided convolution."""
    def __init__(self, channels: int):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling module using nearest-neighbor interpolation followed by convolution."""
    def __init__(self, channels: int):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class MeDiUNet(nn.Module):
    """
    The core UNet architecture for the MeDi baseline.
    Processes the noisy image x_t conditioned on the single fused vector z_final.
    """
    
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3, 
                 base_channels: int = 64, 
                 channel_multipliers: List[int] = [1, 2, 4], 
                 cond_dim: int = 256,
                 dropout: float = 0.1):
        """
        :param in_channels: Number of input image channels.
        :param out_channels: Number of output channels.
        :param base_channels: Base number of channels in the UNet.
        :param channel_multipliers: Channel multipliers for each resolution level.
        :param cond_dim: Dimension of the fused conditioning vector (z_final).
        :param dropout: Dropout rate for ResidualBlocks.
        """
        super(MeDiUNet, self).__init__()
        
        # Initial Convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsampling Path (Encoder)
        self.downs = nn.ModuleList()
        channels = [base_channels]
        current_channels = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            self.downs.append(nn.ModuleList([
                MeDiResidualBlock(current_channels, out_ch, cond_dim, dropout),
                MeDiResidualBlock(out_ch, out_ch, cond_dim, dropout)
            ]))
            channels.extend([out_ch, out_ch])
            current_channels = out_ch
            
            if i != len(channel_multipliers) - 1:
                self.downs.append(nn.ModuleList([Downsample(current_channels), None]))
                channels.append(current_channels)
                
        # Middle Block
        mid_channels = base_channels * channel_multipliers[-1]
        self.mid_block1 = MeDiResidualBlock(mid_channels, mid_channels, cond_dim, dropout)
        self.mid_block2 = MeDiResidualBlock(mid_channels, mid_channels, cond_dim, dropout)
        
        # Upsampling Path (Decoder)
        self.ups = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            in_ch = current_channels + channels.pop()
            
            self.ups.append(nn.ModuleList([
                MeDiResidualBlock(in_ch, out_ch, cond_dim, dropout),
                MeDiResidualBlock(out_ch, out_ch, cond_dim, dropout)
            ]))
            current_channels = out_ch
            channels.pop()
            channels.pop()
            
            if i != 0:
                self.ups.append(nn.ModuleList([Upsample(current_channels), None]))
                
        # Final Output Convolution
        self.final_norm = nn.GroupNorm(num_groups=min(32, current_channels), num_channels=current_channels)
        self.final_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes the final convolution layer with zeros for stable early training."""
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: torch.Tensor, z_final: torch.Tensor) -> torch.Tensor:
        """
        :param x: Noisy input image of shape (B, C_in, H, W).
        :param z_final: Fused conditioning vector of shape (B, cond_dim).
        :return: Predicted noise of shape (B, C_out, H, W).
        """
        h = self.init_conv(x)
        h_stack = [h]
        
        # Downsampling path
        for block_group in self.downs:
            b1, b2 = block_group[0], block_group[1]
            if b2 is None: 
                h = b1(h)
            else:
                h = b1(h, z_final)
                h_stack.append(h)
                h = b2(h, z_final)
                h_stack.append(h)
                
        # Middle blocks
        h = self.mid_block1(h, z_final)
        h = self.mid_block2(h, z_final)
        
        # Upsampling path with skip connections
        for block_group in self.ups:
            b1, b2 = block_group[0], block_group[1]
            if b2 is None:
                h = b1(h)
            else:
                h = torch.cat([h, h_stack.pop()], dim=1)
                h = b1(h, z_final)
                h = torch.cat([h, h_stack.pop()], dim=1)
                h = b2(h, z_final)
                
        # Final output
        h = self.final_norm(h)
        h = F.silu(h)
        out = self.final_conv(h)
        
        return out


class MeDi(nn.Module):
    """
    Full MeDi (Metadata-guided Diffusion) baseline model.
    Integrates discrete class/metadata embeddings and additive fusion.
    """
    
    def __init__(self,
                 num_classes: int,
                 metadata_info: List[Tuple[str, int]],
                 in_channels: int = 3,
                 out_channels: int = 3,
                 base_channels: int = 64,
                 channel_multipliers: List[int] = [1, 2, 4],
                 time_emb_dim: int = 256,
                 embed_dim: int = 256,
                 dropout: float = 0.1):
        """
        :param num_classes: Number of discrete biological classes (K).
        :param metadata_info: List of tuples (metadata_name, num_categories).
        :param in_channels: Number of input image channels.
        :param out_channels: Number of output channels.
        :param base_channels: Base number of channels in the UNet.
        :param channel_multipliers: Channel multipliers for each resolution level.
        :param time_emb_dim: Dimension of the time embedding.
        :param embed_dim: Dimension of the discrete embeddings and fused conditioning vector.
        :param dropout: Dropout rate for regularization.
        """
        super(MeDi, self).__init__()
        
        if num_classes <= 0:
            raise ValueError("num_classes must be strictly positive.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be strictly positive.")
            
        self.embed_dim = embed_dim
        
        # 1. Discrete Embeddings: E_class(y) + sum E_meta^{(j)}(m_j)
        # Reuses the DiscreteEmbedding module from the conditioning package
        self.discrete_cond = DiscreteEmbedding(
            num_classes=num_classes, 
            metadata_info=metadata_info, 
            embed_dim=embed_dim
        )
        
        # 2. Time Embedding MLP: z_t = MLP(t)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, embed_dim) # Project to embed_dim to match z_cond
        )
        
        # 3. UNet Backbone
        self.unet = MeDiUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            cond_dim=embed_dim,
            dropout=dropout
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes the time MLP for stable gradient flow."""
        for m in self.time_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the MeDi baseline model.
        
        Mathematical Formulation:
        1. z_cond = E_class(y) + sum_{j=1}^{J} E_meta^{(j)}(m_j)
        2. z_t = MLP(t)
        3. z_final = z_t + z_cond  (Additive Fusion)
        4. epsilon_pred = UNet(x_t, z_final)
        
        :param x_t: Noisy input image of shape (B, C_in, H, W).
        :param t: Diffusion timestep of shape (B,).
        :param y: Discrete class label of shape (B,).
        :param metadata: Dictionary of discrete metadata tokens.
        :return: Predicted noise epsilon_pred of shape (B, C_out, H, W).
        """
        if x_t.dim() != 4:
            raise ValueError(f"Expected input image x_t to be 4D (B, C, H, W), got {x_t.dim()}D.")
        if t.dim() != 1:
            raise ValueError(f"Expected timestep t to be 1D (B,), got {t.dim()}D.")
            
        # 1. Compute Additive Conditioning: z_cond = E_class(y) + sum E_meta(m_j)
        z_cond = self.discrete_cond(y, metadata)
        
        # 2. Compute Time Embedding: z_t = MLP(t)
        z_t = self.time_mlp(t.float())
        
        # 3. Additive Fusion: z_final = z_t + z_cond
        # This linear summation is the core flaw that causes morphological entanglement
        z_final = z_t + z_cond
        
        # 4. Predict Noise via UNet Backbone
        epsilon_pred = self.unet(x_t, z_final)
        
        return epsilon_pred


if __name__ == "__main__":
    # Example usage and validation of the MeDi baseline model
    
    # 1. Define hyperparameters
    batch_size = 4
    num_classes = 32
    in_channels = 3
    out_channels = 3
    height, width = 64, 64
    base_channels = 32
    channel_multipliers = [1, 2, 4]
    time_emb_dim = 128
    embed_dim = 128
    
    metadata_info = [
        ('tissue_source_site', 40),
        ('scanner_id', 5)
    ]
    
    # 2. Instantiate the MeDi model
    model = MeDi(
        num_classes=num_classes,
        metadata_info=metadata_info,
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        time_emb_dim=time_emb_dim,
        embed_dim=embed_dim,
        dropout=0.1
    )
    
    print(f"Initialized MeDi Baseline:")
    print(f"  Number of Classes (K): {num_classes}")
    print(f"  Metadata Tokens: {[name for name, _ in metadata_info]}")
    print(f"  Embedding Dimension (d): {embed_dim}")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Create orig inputs
    x_t_orig = torch.randn(batch_size, in_channels, height, width)
    t_orig = torch.randint(0, 1000, (batch_size,))
    y_orig = torch.randint(0, num_classes, (batch_size,))
    metadata_orig = {
        'tissue_source_site': torch.randint(0, 40, (batch_size,)),
        'scanner_id': torch.randint(0, 5, (batch_size,))
    }
    
    # 4. Forward pass
    epsilon_pred = model(x_t_orig, t_orig, y_orig, metadata_orig)
    
    print(f"\nForward Pass Validation:")
    print(f"  Input x_t shape: {x_t_orig.shape}")
    print(f"  Output epsilon_pred shape: {epsilon_pred.shape}")
    
    # 5. Verify output shape
    assert epsilon_pred.shape == (batch_size, out_channels, height, width), "Output shape mismatch!"
    
    # 6. Test backward pass
    loss = epsilon_pred.sum()
    loss.backward()
    
    # Check if gradients flowed to the discrete embeddings and UNet
    assert model.discrete_cond.E_class.weight.grad is not None, "Gradients did not flow to E_class!"
    assert model.unet.init_conv.weight.grad is not None, "Gradients did not flow to UNet!"
    
    print("\nVerification passed: MeDi shapes are correct and gradients flow successfully through the additive conditioning pipeline.")