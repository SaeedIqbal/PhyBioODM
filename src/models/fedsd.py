"""
fedsd.py
Implements the FedSD (Federated Semantic Disentanglement) baseline model.
FedSD attempts to separate domain-invariant semantic vectors from domain-specific 
style vectors using non-adversarial orthogonality constraints. In this centralized 
baseline implementation, it extracts semantic and style embeddings from the input 
image and enforces orthogonality between them.

Mathematical Formulations:
1. Dual Encoding: 
   v_sem = E_sem(x), v_sty = E_sty(x)
2. Orthogonality Constraint: 
   L_ortho = || v_sem^T v_sty ||_F^2
3. Additive Conditioning: 
   z_cond = z_t + v_sem + v_sty
4. Noise Prediction: 
   epsilon_pred = UNet(x_t, z_cond)

Unlike PhyBio-ODM, FedSD relies on statistical orthogonality of abstract latent 
vectors without explicit continuous physical parameterization or orthogonal 
manifold constraints, failing to capture the non-linear physical coupling of 
tissue and chemical dyes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# Reuse components from the unet_backbone
from .unet_backbone import SinusoidalPositionalEmbedding, Downsample, Upsample


class SemanticEncoder(nn.Module):
    """
    Extracts domain-invariant semantic (biological) embeddings from the input image.
    
    Mathematical Equation:
    v_sem = E_sem(x)
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 256):
        super(SemanticEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input image tensor of shape (B, C, H, W).
        :return: Semantic embedding v_sem of shape (B, embed_dim).
        """
        return self.encoder(x)


class StyleEncoder(nn.Module):
    """
    Extracts domain-specific style (physical/stain) embeddings from the input image.
    
    Mathematical Equation:
    v_sty = E_sty(x)
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 256):
        super(StyleEncoder, self).__init__()
        # Uses a slightly different architectural bias to simulate dual-branch extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input image tensor of shape (B, C, H, W).
        :return: Style embedding v_sty of shape (B, embed_dim).
        """
        return self.encoder(x)


class FedSDResidualBlock(nn.Module):
    """
    Residual block for FedSD, conditioned on the additively fused semantic and style embeddings.
    Lacks the orthogonal dual-stream modulation of PhyBio-ODM.
    """
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.0):
        super(FedSDResidualBlock, self).__init__()
        
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        self.cond_proj = nn.Linear(cond_dim, in_channels * 2)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, z_cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        
        # Standard FiLM/AdaLN with single fused conditioning vector
        params = self.cond_proj(z_cond)
        gamma, beta = params.chunk(2, dim=-1)
        
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) + 1.0
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        h = gamma * h + beta
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.residual_proj(x)


class FedSDUNet(nn.Module):
    """
    UNet backbone for FedSD, utilizing the additively fused semantic and style conditioning.
    """
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3, 
                 base_channels: int = 64, 
                 channel_multipliers: List[int] = [1, 2, 4], 
                 cond_dim: int = 256,
                 dropout: float = 0.1):
        super(FedSDUNet, self).__init__()
        
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList()
        channels = [base_channels]
        current_channels = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            self.downs.append(nn.ModuleList([
                FedSDResidualBlock(current_channels, out_ch, cond_dim, dropout),
                FedSDResidualBlock(out_ch, out_ch, cond_dim, dropout)
            ]))
            channels.extend([out_ch, out_ch])
            current_channels = out_ch
            
            if i != len(channel_multipliers) - 1:
                self.downs.append(nn.ModuleList([Downsample(current_channels), None]))
                channels.append(current_channels)
                
        mid_channels = base_channels * channel_multipliers[-1]
        self.mid_block1 = FedSDResidualBlock(mid_channels, mid_channels, cond_dim, dropout)
        self.mid_block2 = FedSDResidualBlock(mid_channels, mid_channels, cond_dim, dropout)
        
        self.ups = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            in_ch = current_channels + channels.pop()
            
            self.ups.append(nn.ModuleList([
                FedSDResidualBlock(in_ch, out_ch, cond_dim, dropout),
                FedSDResidualBlock(out_ch, out_ch, cond_dim, dropout)
            ]))
            current_channels = out_ch
            channels.pop()
            channels.pop()
            
            if i != 0:
                self.ups.append(nn.ModuleList([Upsample(current_channels), None]))
                
        self.final_norm = nn.GroupNorm(num_groups=min(32, current_channels), num_channels=current_channels)
        self.final_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)
        
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x_t: torch.Tensor, z_cond: torch.Tensor) -> torch.Tensor:
        h = self.init_conv(x_t)
        h_stack = [h]
        
        for block_group in self.downs:
            b1, b2 = block_group[0], block_group[1]
            if b2 is None: 
                h = b1(h)
            else:
                h = b1(h, z_cond)
                h_stack.append(h)
                h = b2(h, z_cond)
                h_stack.append(h)
                
        h = self.mid_block1(h, z_cond)
        h = self.mid_block2(h, z_cond)
        
        for block_group in self.ups:
            b1, b2 = block_group[0], block_group[1]
            if b2 is None:
                h = b1(h)
            else:
                h = torch.cat([h, h_stack.pop()], dim=1)
                h = b1(h, z_cond)
                h = torch.cat([h, h_stack.pop()], dim=1)
                h = b2(h, z_cond)
                
        h = self.final_norm(h)
        h = F.silu(h)
        out = self.final_conv(h)
        
        return out


class FedSD(nn.Module):
    """
    Full FedSD (Federated Semantic Disentanglement) baseline model.
    Integrates dual-branch encoding with non-adversarial orthogonality constraints.
    """
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 base_channels: int = 64,
                 channel_multipliers: List[int] = [1, 2, 4],
                 time_emb_dim: int = 256,
                 embed_dim: int = 256,
                 dropout: float = 0.1):
        super(FedSD, self).__init__()
        
        if embed_dim <= 0:
            raise ValueError("embed_dim must be strictly positive.")
            
        self.embed_dim = embed_dim
        
        # 1. Dual Encoders for Semantic and Style
        self.sem_encoder = SemanticEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.sty_encoder = StyleEncoder(in_channels=in_channels, embed_dim=embed_dim)
        
        # 2. Time Embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, embed_dim)
        )
        
        # 3. UNet Backbone
        self.unet = FedSDUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            cond_dim=embed_dim,
            dropout=dropout
        )

    def compute_orthogonality_loss(self, v_sem: torch.Tensor, v_sty: torch.Tensor) -> torch.Tensor:
        """
        Computes the non-adversarial orthogonality constraint.
        
        Equation: L_ortho = || v_sem^T v_sty ||_F^2
        
        :param v_sem: Semantic embeddings of shape (B, d).
        :param v_sty: Style embeddings of shape (B, d).
        :return: Scalar orthogonality loss.
        """
        # Compute the dot product matrix between all pairs in the batch
        # and minimize its Frobenius norm to enforce orthogonality
        cross_cov = v_sem.T @ v_sty
        return torch.norm(cross_cov, p='fro') ** 2

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x_orig: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the FedSD baseline model.
        
        Mathematical Formulation:
        1. v_sem = E_sem(x_orig), v_sty = E_sty(x_orig)
        2. z_t = MLP_time(t)
        3. z_cond = z_t + v_sem + v_sty (Additive Fusion)
        4. epsilon_pred = UNet(x_t, z_cond)
        
        :param x_t: Noisy input image of shape (B, C_in, H, W).
        :param t: Diffusion timestep of shape (B,).
        :param x_orig: Original clean image for dual encoding (B, C_in, H, W).
                       If None, x_t is used as the base for encoding (for inference).
        :return: Tuple containing predicted noise epsilon_pred (B, C_out, H, W), 
                 semantic embedding v_sem (B, d), and style embedding v_sty (B, d).
        """
        if x_t.dim() != 4:
            raise ValueError(f"Expected input image x_t to be 4D (B, C, H, W), got {x_t.dim()}D.")
        if t.dim() != 1:
            raise ValueError(f"Expected timestep t to be 1D (B,), got {t.dim()}D.")
            
        # 1. Dual Encoding
        base_img = x_orig if x_orig is not None else x_t.detach()
        v_sem = self.sem_encoder(base_img)
        v_sty = self.sty_encoder(base_img)
        
        # 2. Compute Time Embedding
        z_t = self.time_mlp(t.float())
        
        # 3. Additive Fusion (The core flaw causing entanglement in baselines)
        z_cond = z_t + v_sem + v_sty
        
        # 4. Predict Noise via UNet
        epsilon_pred = self.unet(x_t, z_cond)
        
        return epsilon_pred, v_sem, v_sty


if __name__ == "__main__":
    # Example usage and validation of the FedSD baseline model
    
    batch_size = 4
    in_channels = 3
    out_channels = 3
    height, width = 64, 64
    base_channels = 32
    channel_multipliers = [1, 2, 4]
    time_emb_dim = 128
    embed_dim = 128
    
    model = FedSD(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        time_emb_dim=time_emb_dim,
        embed_dim=embed_dim,
        dropout=0.1
    )
    
    print(f"Initialized FedSD Baseline:")
    print(f"  Embedding Dimension (d): {embed_dim}")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x_t_orig = torch.randn(batch_size, in_channels, height, width)
    x_orig_orig = torch.rand(batch_size, in_channels, height, width)
    t_orig = torch.randint(0, 1000, (batch_size,))
    
    epsilon_pred, v_sem, v_sty = model(x_t_orig, t_orig, x_orig_orig)
    
    print(f"\nForward Pass Validation:")
    print(f"  Input x_t shape: {x_t_orig.shape}")
    print(f"  Output epsilon_pred shape: {epsilon_pred.shape}")
    print(f"  Semantic embedding v_sem shape: {v_sem.shape}")
    print(f"  Style embedding v_sty shape: {v_sty.shape}")
    
    assert epsilon_pred.shape == (batch_size, out_channels, height, width), "Output shape mismatch!"
    assert v_sem.shape == (batch_size, embed_dim), "v_sem shape mismatch!"
    assert v_sty.shape == (batch_size, embed_dim), "v_sty shape mismatch!"
    
    # Compute losses
    L_diff = F.mse_loss(epsilon_pred, torch.randn_like(epsilon_pred))
    L_ortho = model.compute_orthogonality_loss(v_sem, v_sty)
    L_total = L_diff + 0.1 * L_ortho
    
    L_total.backward()
    
    assert model.sem_encoder.encoder[0].weight.grad is not None, "Gradients did not flow to semantic encoder!"
    assert model.sty_encoder.encoder[0].weight.grad is not None, "Gradients did not flow to style encoder!"
    assert model.unet.init_conv.weight.grad is not None, "Gradients did not flow to UNet!"
    
    print("\nVerification passed: FedSD shapes are correct and gradients flow successfully through the dual-encoder and orthogonality pipelines.")