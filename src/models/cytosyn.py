"""
cytosyn.py
Implements the CytoSyn (Foundation Diffusion) baseline model.
CytoSyn represents a large-scale foundation diffusion approach that models 
visual variations using a unified, coupled conditioning space. 
Unlike PhyBio-ODM's orthogonal dual-stream architecture, CytoSyn forces 
all conditioning factors (biology, metadata, style) into a single shared 
latent representation via concatenation and a shared MLP. This coupled 
approach lacks explicit physical-biological separation, leading to 
entangled feature spaces and potential morphological distortion.

Mathematical Formulations:
1. Class Embedding: v_class = E_class(y)
2. Metadata Embeddings: v_meta_j = E_meta_j(m_j)
3. Coupled Conditioning: v_coupled = MLP_shared([v_class || v_meta_1 || ... || v_meta_J])
4. Time Embedding: z_t = MLP_time(t)
5. Final Conditioning: z_final = z_t + v_coupled
6. AdaLN Modulation: h_next = gamma(z_final) * LayerNorm(h) + beta(z_final)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

# Reuse components from the unet_backbone to maintain consistency and reduce redundancy
from .unet_backbone import SinusoidalPositionalEmbedding, Downsample, Upsample


class CoupledConditioning(nn.Module):
    """
    Implements the unified, coupled conditioning space for CytoSyn.
    Concatenates all discrete embeddings and projects them through a shared MLP,
    forcing biological and physical factors into a single entangled latent vector.
    """
    
    def __init__(self, num_classes: int, metadata_info: List[Tuple[str, int]], embed_dim: int):
        """
        :param num_classes: Number of discrete biological classes (K).
        :param metadata_info: List of tuples (metadata_name, num_categories).
        :param embed_dim: Dimension of the individual embeddings and the final coupled vector.
        """
        super(CoupledConditioning, self).__init__()
        
        if num_classes <= 0 or embed_dim <= 0:
            raise ValueError("num_classes and embed_dim must be strictly positive.")
            
        self.embed_dim = embed_dim
        
        # Individual embedding tables
        self.E_class = nn.Embedding(num_embeddings=num_classes, embedding_dim=embed_dim)
        self.E_meta = nn.ModuleDict()
        for meta_name, num_categories in metadata_info:
            self.E_meta[meta_name] = nn.Embedding(num_embeddings=num_categories, embedding_dim=embed_dim)
            
        # Shared MLP for coupled conditioning
        # Input dimension is the concatenation of all embeddings
        concat_dim = embed_dim * (1 + len(metadata_info))
        
        self.mlp_shared = nn.Sequential(
            nn.Linear(concat_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes embedding tables and MLP layers."""
        nn.init.normal_(self.E_class.weight, mean=0.0, std=0.02)
        for meta_name in self.E_meta:
            nn.init.normal_(self.E_meta[meta_name].weight, mean=0.0, std=0.02)
            
        for m in self.mlp_shared.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, y: torch.Tensor, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the coupled conditioning vector.
        
        Equation: v_coupled = MLP_shared([E_class(y) || E_meta_1(m_1) || ... || E_meta_J(m_J)])
        
        :param y: Discrete class labels of shape (B,).
        :param metadata: Dictionary of discrete metadata tokens.
        :return: Coupled conditioning vector v_coupled of shape (B, embed_dim).
        """
        batch_size = y.shape[0]
        
        # 1. Compute individual embeddings
        v_class = self.E_class(y)
        v_meta_list = [self.E_meta[name](metadata[name]) for name in metadata]
        
        # 2. Concatenate all embeddings: [v_class || v_meta_1 || ... || v_meta_J]
        v_concat = torch.cat([v_class] + v_meta_list, dim=-1)
        
        # 3. Project through shared MLP to get coupled vector
        v_coupled = self.mlp_shared(v_concat)
        
        return v_coupled


class CytoSynResidualBlock(nn.Module):
    """
    Residual block for CytoSyn using standard single-stream AdaLN.
    Lacks the orthogonal dual-stream modulation of PhyBio-ODM.
    
    Mathematical Formulation:
    h_next = gamma(z_final) * LayerNorm(h) + beta(z_final)
    """
    
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.0):
        super(CytoSynResidualBlock, self).__init__()
        
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        self.cond_proj = nn.Linear(cond_dim, in_channels * 2)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, z_final: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        
        # Single-stream AdaLN using the coupled vector
        params = self.cond_proj(z_final)
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


class CytoSynUNet(nn.Module):
    """
    UNet backbone for CytoSyn, utilizing the coupled conditioning space.
    """
    
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3, 
                 base_channels: int = 64, 
                 channel_multipliers: List[int] = [1, 2, 4], 
                 cond_dim: int = 256,
                 dropout: float = 0.1):
        super(CytoSynUNet, self).__init__()
        
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList()
        channels = [base_channels]
        current_channels = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            self.downs.append(nn.ModuleList([
                CytoSynResidualBlock(current_channels, out_ch, cond_dim, dropout),
                CytoSynResidualBlock(out_ch, out_ch, cond_dim, dropout)
            ]))
            channels.extend([out_ch, out_ch])
            current_channels = out_ch
            
            if i != len(channel_multipliers) - 1:
                self.downs.append(nn.ModuleList([Downsample(current_channels), None]))
                channels.append(current_channels)
                
        mid_channels = base_channels * channel_multipliers[-1]
        self.mid_block1 = CytoSynResidualBlock(mid_channels, mid_channels, cond_dim, dropout)
        self.mid_block2 = CytoSynResidualBlock(mid_channels, mid_channels, cond_dim, dropout)
        
        self.ups = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            in_ch = current_channels + channels.pop()
            
            self.ups.append(nn.ModuleList([
                CytoSynResidualBlock(in_ch, out_ch, cond_dim, dropout),
                CytoSynResidualBlock(out_ch, out_ch, cond_dim, dropout)
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

    def forward(self, x: torch.Tensor, z_final: torch.Tensor) -> torch.Tensor:
        h = self.init_conv(x)
        h_stack = [h]
        
        for block_group in self.downs:
            b1, b2 = block_group[0], block_group[1]
            if b2 is None: 
                h = b1(h)
            else:
                h = b1(h, z_final)
                h_stack.append(h)
                h = b2(h, z_final)
                h_stack.append(h)
                
        h = self.mid_block1(h, z_final)
        h = self.mid_block2(h, z_final)
        
        for block_group in self.ups:
            b1, b2 = block_group[0], block_group[1]
            if b2 is None:
                h = b1(h)
            else:
                h = torch.cat([h, h_stack.pop()], dim=1)
                h = b1(h, z_final)
                h = torch.cat([h, h_stack.pop()], dim=1)
                h = b2(h, z_final)
                
        h = self.final_norm(h)
        h = F.silu(h)
        out = self.final_conv(h)
        
        return out


class CytoSyn(nn.Module):
    """
    Full CytoSyn (Foundation Diffusion) baseline model.
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
        super(CytoSyn, self).__init__()
        
        if num_classes <= 0 or embed_dim <= 0:
            raise ValueError("num_classes and embed_dim must be strictly positive.")
            
        self.embed_dim = embed_dim
        
        # 1. Coupled Conditioning Space
        self.coupled_cond = CoupledConditioning(
            num_classes=num_classes, 
            metadata_info=metadata_info, 
            embed_dim=embed_dim
        )
        
        # 2. Time Embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, embed_dim)
        )
        
        # 3. UNet Backbone
        self.unet = CytoSynUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            cond_dim=embed_dim,
            dropout=dropout
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the CytoSyn baseline model.
        
        Mathematical Formulation:
        1. v_coupled = MLP_shared([E_class(y) || E_meta(m)])
        2. z_t = MLP_time(t)
        3. z_final = z_t + v_coupled (Coupled Additive Fusion)
        4. epsilon_pred = UNet(x_t, z_final)
        """
        if x_t.dim() != 4:
            raise ValueError(f"Expected input image x_t to be 4D (B, C, H, W), got {x_t.dim()}D.")
        if t.dim() != 1:
            raise ValueError(f"Expected timestep t to be 1D (B,), got {t.dim()}D.")
            
        # 1. Compute Coupled Conditioning
        v_coupled = self.coupled_cond(y, metadata)
        
        # 2. Compute Time Embedding
        z_t = self.time_mlp(t.float())
        
        # 3. Additive Fusion (Coupled)
        z_final = z_t + v_coupled
        
        # 4. Predict Noise
        epsilon_pred = self.unet(x_t, z_final)
        
        return epsilon_pred


if __name__ == "__main__":
    # Example usage and validation of the CytoSyn baseline model
    
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
    
    model = CytoSyn(
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
    
    print(f"Initialized CytoSyn Baseline:")
    print(f"  Number of Classes (K): {num_classes}")
    print(f"  Metadata Tokens: {[name for name, _ in metadata_info]}")
    print(f"  Coupled Embedding Dimension (d): {embed_dim}")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x_t_orig = torch.randn(batch_size, in_channels, height, width)
    t_orig = torch.randint(0, 1000, (batch_size,))
    y_orig = torch.randint(0, num_classes, (batch_size,))
    metadata_orig = {
        'tissue_source_site': torch.randint(0, 40, (batch_size,)),
        'scanner_id': torch.randint(0, 5, (batch_size,))
    }
    
    epsilon_pred = model(x_t_orig, t_orig, y_orig, metadata_orig)
    
    print(f"\nForward Pass Validation:")
    print(f"  Input x_t shape: {x_t_orig.shape}")
    print(f"  Output epsilon_pred shape: {epsilon_pred.shape}")
    
    assert epsilon_pred.shape == (batch_size, out_channels, height, width), "Output shape mismatch!"
    
    loss = epsilon_pred.sum()
    loss.backward()
    
    assert model.coupled_cond.E_class.weight.grad is not None, "Gradients did not flow to E_class!"
    assert model.unet.init_conv.weight.grad is not None, "Gradients did not flow to UNet!"
    
    print("\nVerification passed: CytoSyn shapes are correct and gradients flow successfully through the coupled conditioning pipeline.")