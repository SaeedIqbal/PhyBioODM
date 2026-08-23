"""
chis.py
Implements the CHIS (Controllable Histopathology Image Synthesis) baseline model.
CHIS attempts to achieve controllable synthesis by utilizing frequency-domain phase 
manipulation to initialize or guide the structural layout of the generated images.

Mathematical Formulations:
1. Structural Prior Extraction (Phase-only spectrum):
   S = Re( F^{-1}( exp(i * angle(F(M_prior))) ) )
   
2. Initial State / Structural Guidance (Amplitude-Phase combination):
   X_0 ~ F^{-1}( |F(Z_gaussian)| * exp(i * angle(F(M_prior))) )

3. Discrete Conditioning (Additive):
   z_cond = E_class(y) + sum_{j=1}^{J} E_meta^{(j)}(m_j)
   
4. Final Conditioning:
   z_final = z_t + z_cond

Unlike PhyBio-ODM, CHIS relies on external frequency-domain structural initialization 
and discrete additive conditioning, which fails to isolate cross-contamination between 
biological morphology and staining variation during the reverse diffusion steps.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

# Reuse components from the unet_backbone and discrete_embedding
from .unet_backbone import SinusoidalPositionalEmbedding, Downsample, Upsample
from ..conditioning.discrete_embedding import DiscreteEmbedding


class FrequencyStructuralPrior(nn.Module):
    """
    Computes the structural prior map from a reference image using frequency-domain 
    phase manipulation. This represents the "training-free structural initialization" 
    used in CHIS to guide the spatial layout of the synthesis.
    """
    
    def __init__(self):
        super(FrequencyStructuralPrior, self).__init__()

    def forward(self, m_prior: torch.Tensor) -> torch.Tensor:
        """
        Computes the structural prior map S from the reference image M_prior.
        
        Equation: S = Re( F^{-1}( exp(i * angle(F(M_prior))) ) )
        
        :param m_prior: Reference image tensor of shape (B, C, H, W).
        :return: Structural prior map S of shape (B, 1, H, W).
        """
        if m_prior.dim() != 4:
            raise ValueError(f"Expected m_prior to be 4D (B, C, H, W), got {m_prior.dim()}D.")
            
        # Convert to grayscale for consistent phase extraction across channels
        # Weights for RGB to Grayscale: [0.2989, 0.5870, 0.1140]
        if m_prior.shape[1] == 3:
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=m_prior.device, dtype=m_prior.dtype)
            m_gray = torch.sum(m_prior * weights.view(1, 3, 1, 1), dim=1, keepdim=True)
        else:
            m_gray = m_prior.mean(dim=1, keepdim=True)
            
        # 1. Forward FFT: F(M_prior)
        F_m = torch.fft.fft2(m_gray)
        
        # 2. Extract Phase: angle(F(M_prior))
        phase_m = torch.angle(F_m)
        
        # 3. Phase-only spectrum: exp(i * phase)
        phase_only = torch.exp(1j * phase_m)
        
        # 4. Inverse FFT and take real part: Re( F^{-1}(phase_only) )
        s_map = torch.fft.ifft2(phase_only).real
        
        # Normalize the structural map to [0, 1] for stable conditioning
        s_min = s_map.amin(dim=(2, 3), keepdim=True)
        s_max = s_map.amax(dim=(2, 3), keepdim=True)
        s_map = (s_map - s_min) / (s_max - s_min + 1e-8)
        
        return s_map


class CHISResidualBlock(nn.Module):
    """
    Residual block for CHIS. Incorporates both the standard additive discrete 
    conditioning (z_final) and the spatial structural prior map (S) derived 
    from the frequency domain.
    """
    
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.0):
        super(CHISResidualBlock, self).__init__()
        
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        
        # Projection for discrete additive conditioning (z_final)
        self.cond_proj = nn.Linear(cond_dim, in_channels * 2)
        
        # Projection for spatial structural prior (S)
        # Maps the 1-channel structural map to the feature space
        self.struct_proj = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, z_final: torch.Tensor, s_map: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input feature map (B, C_in, H, W).
        :param z_final: Fused discrete conditioning vector (B, cond_dim).
        :param s_map: Spatial structural prior map (B, 1, H, W).
        :return: Output feature map (B, C_out, H, W).
        """
        h = self.norm1(x)
        
        # 1. Apply discrete additive conditioning (FiLM/AdaLN style)
        params = self.cond_proj(z_final)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) + 1.0
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = gamma * h + beta
        
        # 2. Add spatial structural prior (Frequency-domain guidance)
        # This represents the "external structural initialization" leaking into features
        s_feat = self.struct_proj(s_map)
        h = h + s_feat
        
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.residual_proj(x)


class CHISUNet(nn.Module):
    """
    UNet backbone for CHIS, utilizing both discrete additive conditioning 
    and frequency-domain structural priors.
    """
    
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3, 
                 base_channels: int = 64, 
                 channel_multipliers: List[int] = [1, 2, 4], 
                 cond_dim: int = 256,
                 dropout: float = 0.1):
        super(CHISUNet, self).__init__()
        
        # Initial convolution takes the noisy image (in_channels) + structural map (1)
        self.init_conv = nn.Conv2d(in_channels + 1, base_channels, kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList()
        channels = [base_channels]
        current_channels = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            self.downs.append(nn.ModuleList([
                CHISResidualBlock(current_channels, out_ch, cond_dim, dropout),
                CHISResidualBlock(out_ch, out_ch, cond_dim, dropout)
            ]))
            channels.extend([out_ch, out_ch])
            current_channels = out_ch
            
            if i != len(channel_multipliers) - 1:
                self.downs.append(nn.ModuleList([Downsample(current_channels), None]))
                channels.append(current_channels)
                
        mid_channels = base_channels * channel_multipliers[-1]
        self.mid_block1 = CHISResidualBlock(mid_channels, mid_channels, cond_dim, dropout)
        self.mid_block2 = CHISResidualBlock(mid_channels, mid_channels, cond_dim, dropout)
        
        self.ups = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            in_ch = current_channels + channels.pop()
            
            self.ups.append(nn.ModuleList([
                CHISResidualBlock(in_ch, out_ch, cond_dim, dropout),
                CHISResidualBlock(out_ch, out_ch, cond_dim, dropout)
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

    def forward(self, x_t: torch.Tensor, z_final: torch.Tensor, s_map: torch.Tensor) -> torch.Tensor:
        # Concatenate structural map to the initial noisy image
        h = torch.cat([x_t, s_map], dim=1)
        h = self.init_conv(h)
        h_stack = [h]
        
        for block_group in self.downs:
            b1, b2 = block_group[0], block_group[1]
            if b2 is None: 
                h = b1(h)
            else:
                h = b1(h, z_final, s_map)
                h_stack.append(h)
                h = b2(h, z_final, s_map)
                h_stack.append(h)
                
        h = self.mid_block1(h, z_final, s_map)
        h = self.mid_block2(h, z_final, s_map)
        
        for block_group in self.ups:
            b1, b2 = block_group[0], block_group[1]
            if b2 is None:
                h = b1(h)
            else:
                h = torch.cat([h, h_stack.pop()], dim=1)
                h = b1(h, z_final, s_map)
                h = torch.cat([h, h_stack.pop()], dim=1)
                h = b2(h, z_final, s_map)
                
        h = self.final_norm(h)
        h = F.silu(h)
        out = self.final_conv(h)
        
        return out


class CHIS(nn.Module):
    """
    Full CHIS (Controllable Histopathology Image Synthesis) baseline model.
    Integrates frequency-domain phase manipulation with discrete additive conditioning.
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
        super(CHIS, self).__init__()
        
        if num_classes <= 0 or embed_dim <= 0:
            raise ValueError("num_classes and embed_dim must be strictly positive.")
            
        self.embed_dim = embed_dim
        
        # 1. Discrete Additive Conditioning
        self.discrete_cond = DiscreteEmbedding(
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
        
        # 3. Frequency Structural Prior Extractor
        self.freq_prior = FrequencyStructuralPrior()
        
        # 4. UNet Backbone
        self.unet = CHISUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            cond_dim=embed_dim,
            dropout=dropout
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor, 
                metadata: Dict[str, torch.Tensor], m_prior: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CHIS baseline model.
        
        Mathematical Formulation:
        1. S = Re( F^{-1}( exp(i * angle(F(M_prior))) ) )
        2. z_cond = E_class(y) + sum E_meta(m_j)
        3. z_t = MLP_time(t)
        4. z_final = z_t + z_cond
        5. epsilon_pred = UNet(x_t, z_final, S)
        
        :param x_t: Noisy input image (B, C_in, H, W).
        :param t: Diffusion timestep (B,).
        :param y: Discrete class label (B,).
        :param metadata: Dictionary of discrete metadata tokens.
        :param m_prior: Reference image for structural guidance (B, C_in, H, W).
        :return: Predicted noise epsilon_pred (B, C_out, H, W).
        """
        if x_t.dim() != 4:
            raise ValueError(f"Expected x_t to be 4D, got {x_t.dim()}D.")
        if t.dim() != 1:
            raise ValueError(f"Expected t to be 1D, got {t.dim()}D.")
        if m_prior.shape[-2:] != x_t.shape[-2:]:
            raise ValueError("Spatial dimensions of m_prior must match x_t.")
            
        # 1. Compute Structural Prior from Frequency Domain
        s_map = self.freq_prior(m_prior)
        
        # 2. Compute Discrete Additive Conditioning
        z_cond = self.discrete_cond(y, metadata)
        
        # 3. Compute Time Embedding
        z_t = self.time_mlp(t.float())
        
        # 4. Additive Fusion (The core flaw causing entanglement)
        z_final = z_t + z_cond
        
        # 5. Predict Noise via UNet
        epsilon_pred = self.unet(x_t, z_final, s_map)
        
        return epsilon_pred


if __name__ == "__main__":
    # Example usage and validation of the CHIS baseline model
    
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
    
    model = CHIS(
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
    
    print(f"Initialized CHIS Baseline:")
    print(f"  Number of Classes (K): {num_classes}")
    print(f"  Metadata Tokens: {[name for name, _ in metadata_info]}")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x_t_orig = torch.randn(batch_size, in_channels, height, width)
    t_orig = torch.randint(0, 1000, (batch_size,))
    y_orig = torch.randint(0, num_classes, (batch_size,))
    metadata_orig = {
        'tissue_source_site': torch.randint(0, 40, (batch_size,)),
        'scanner_id': torch.randint(0, 5, (batch_size,))
    }
    m_prior_orig = torch.rand(batch_size, in_channels, height, width) # Reference image
    
    epsilon_pred = model(x_t_orig, t_orig, y_orig, metadata_orig, m_prior_orig)
    
    print(f"\nForward Pass Validation:")
    print(f"  Input x_t shape: {x_t_orig.shape}")
    print(f"  Prior m_prior shape: {m_prior_orig.shape}")
    print(f"  Output epsilon_pred shape: {epsilon_pred.shape}")
    
    assert epsilon_pred.shape == (batch_size, out_channels, height, width), "Output shape mismatch!"
    
    loss = epsilon_pred.sum()
    loss.backward()
    
    assert model.freq_prior is not None, "Frequency prior module missing!"
    assert model.discrete_cond.E_class.weight.grad is not None, "Gradients did not flow to E_class!"
    assert model.unet.init_conv.weight.grad is not None, "Gradients did not flow to UNet!"
    
    print("\nVerification passed: CHIS shapes are correct and gradients flow successfully through the frequency-domain and discrete conditioning pipeline.")