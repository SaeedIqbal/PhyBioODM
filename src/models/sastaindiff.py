"""
sastaindiff.py
Implements the SAStainDiff (Self-supervised Stain Diffusion) baseline model.
SAStainDiff embeds stain augmentation directly into the diffusion training loop 
to improve structural robustness. It uses a self-supervised objective to learn 
a continuous stain embedding without relying on discrete metadata tokens, 
addressing domain shift by forcing the network to explicitly model stain variations.

Mathematical Formulations:
1. Stain Augmentation: x_aug = x * delta_scale + delta_shift
2. Stain Embedding Extraction: v_stain = Encoder(x_aug)
3. Self-Supervised Objective: L_SS = || MLP(v_stain) - delta_vec ||_2^2
4. Diffusion Loss: L_diff = || epsilon - epsilon_theta(x_t, t, v_stain) ||_2^2
5. Total Loss: L_total = L_diff + lambda_SS * L_SS
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# Reuse components from the unet_backbone
from .unet_backbone import SinusoidalPositionalEmbedding, Downsample, Upsample


class StainAugmenter(nn.Module):
    """
    Applies random stain perturbations to the input image to simulate 
    domain shifts in staining chemistry and scanner optics.
    
    Mathematical Formulation:
    x_aug = x * delta_scale + delta_shift
    where delta_scale and delta_shift are sampled from uniform distributions.
    """
    
    def __init__(self, perturbation_scale: float = 0.2):
        """
        :param perturbation_scale: Maximum magnitude of the random stain perturbation.
        """
        super(StainAugmenter, self).__init__()
        if perturbation_scale <= 0:
            raise ValueError("perturbation_scale must be strictly positive.")
        self.perturbation_scale = perturbation_scale

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: Input image tensor of shape (B, C, H, W) in [0, 1].
        :return: Tuple containing the augmented image x_aug and the perturbation vector delta_vec.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input x to be 4D (B, C, H, W), got {x.dim()}D.")
            
        B, C, H, W = x.shape
        device = x.device
        
        # Sample random per-channel scaling and shifting factors
        # delta_scale in [1 - scale, 1 + scale]
        delta_scale = 1.0 + (torch.rand(B, C, 1, 1, device=device) - 0.5) * 2 * self.perturbation_scale
        
        # delta_shift in [-scale, scale]
        delta_shift = (torch.rand(B, C, 1, 1, device=device) - 0.5) * 2 * self.perturbation_scale
        
        # Combine into a single perturbation vector for the self-supervised target
        # delta_vec shape: (B, 2 * C)
        delta_vec = torch.cat([
            delta_scale.view(B, C), 
            delta_shift.view(B, C)
        ], dim=1)
        
        # Apply perturbation and clamp to valid image range
        x_aug = x * delta_scale + delta_shift
        x_aug = torch.clamp(x_aug, 0.0, 1.0)
        
        return x_aug, delta_vec


class SelfSupervisedStainEncoder(nn.Module):
    """
    Extracts a continuous stain embedding v_stain from the augmented image 
    and predicts the applied perturbation parameters delta via a self-supervised head.
    
    Mathematical Formulation:
    v_stain = CNN(x_aug)
    delta_pred = MLP(v_stain)
    """
    
    def __init__(self, in_channels: int = 3, embed_dim: int = 256, perturbation_dim: int = 6):
        """
        :param in_channels: Number of input image channels (e.g., 3 for RGB).
        :param embed_dim: Dimension of the extracted stain embedding (d_stain).
        :param perturbation_dim: Dimension of the perturbation vector (2 * C).
        """
        super(SelfSupervisedStainEncoder, self).__init__()
        
        if embed_dim <= 0 or perturbation_dim <= 0:
            raise ValueError("embed_dim and perturbation_dim must be strictly positive.")
            
        self.embed_dim = embed_dim
        
        # CNN Backbone for feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, embed_dim),
            nn.ReLU()
        )
        
        # Self-Supervised Prediction Head
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, perturbation_dim)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_aug: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x_aug: Augmented image tensor of shape (B, C, H, W).
        :return: Tuple containing the stain embedding v_stain (B, embed_dim) 
                 and the predicted perturbation delta_pred (B, perturbation_dim).
        """
        v_stain = self.encoder(x_aug)
        delta_pred = self.pred_head(v_stain)
        return v_stain, delta_pred


class SAStainDiffResidualBlock(nn.Module):
    """
    Residual block for SAStainDiff, conditioned on the continuous stain embedding v_stain.
    Uses standard FiLM/AdaLN modulation since SAStainDiff does not enforce orthogonal dual-stream conditioning.
    """
    
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.0):
        super(SAStainDiffResidualBlock, self).__init__()
        
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        self.cond_proj = nn.Linear(cond_dim, in_channels * 2)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, v_stain: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        
        # FiLM modulation using the stain embedding
        params = self.cond_proj(v_stain)
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


class SAStainDiffUNet(nn.Module):
    """
    UNet backbone for SAStainDiff, conditioned on the continuous stain embedding.
    """
    
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3, 
                 base_channels: int = 64, 
                 channel_multipliers: List[int] = [1, 2, 4], 
                 cond_dim: int = 256,
                 dropout: float = 0.1):
        super(SAStainDiffUNet, self).__init__()
        
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList()
        channels = [base_channels]
        current_channels = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            self.downs.append(nn.ModuleList([
                SAStainDiffResidualBlock(current_channels, out_ch, cond_dim, dropout),
                SAStainDiffResidualBlock(out_ch, out_ch, cond_dim, dropout)
            ]))
            channels.extend([out_ch, out_ch])
            current_channels = out_ch
            
            if i != len(channel_multipliers) - 1:
                self.downs.append(nn.ModuleList([Downsample(current_channels), None]))
                channels.append(current_channels)
                
        mid_channels = base_channels * channel_multipliers[-1]
        self.mid_block1 = SAStainDiffResidualBlock(mid_channels, mid_channels, cond_dim, dropout)
        self.mid_block2 = SAStainDiffResidualBlock(mid_channels, mid_channels, cond_dim, dropout)
        
        self.ups = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            in_ch = current_channels + channels.pop()
            
            self.ups.append(nn.ModuleList([
                SAStainDiffResidualBlock(in_ch, out_ch, cond_dim, dropout),
                SAStainDiffResidualBlock(out_ch, out_ch, cond_dim, dropout)
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

    def forward(self, x_t: torch.Tensor, v_stain: torch.Tensor) -> torch.Tensor:
        h = self.init_conv(x_t)
        h_stack = [h]
        
        for block_group in self.downs:
            b1, b2 = block_group[0], block_group[1]
            if b2 is None: 
                h = b1(h)
            else:
                h = b1(h, v_stain)
                h_stack.append(h)
                h = b2(h, v_stain)
                h_stack.append(h)
                
        h = self.mid_block1(h, v_stain)
        h = self.mid_block2(h, v_stain)
        
        for block_group in self.ups:
            b1, b2 = block_group[0], block_group[1]
            if b2 is None:
                h = b1(h)
            else:
                h = torch.cat([h, h_stack.pop()], dim=1)
                h = b1(h, v_stain)
                h = torch.cat([h, h_stack.pop()], dim=1)
                h = b2(h, v_stain)
                
        h = self.final_norm(h)
        h = F.silu(h)
        out = self.final_conv(h)
        
        return out


class SAStainDiff(nn.Module):
    """
    Full SAStainDiff (Self-supervised Stain Diffusion) baseline model.
    Integrates self-supervised stain augmentation with diffusion-based denoising.
    """
    
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 base_channels: int = 64,
                 channel_multipliers: List[int] = [1, 2, 4],
                 time_emb_dim: int = 256,
                 stain_emb_dim: int = 256,
                 perturbation_scale: float = 0.2,
                 dropout: float = 0.1):
        """
        :param in_channels: Number of input image channels.
        :param out_channels: Number of output channels.
        :param base_channels: Base number of channels in the UNet.
        :param channel_multipliers: Channel multipliers for each resolution level.
        :param time_emb_dim: Dimension of the time embedding.
        :param stain_emb_dim: Dimension of the continuous stain embedding (d_stain).
        :param perturbation_scale: Maximum magnitude of the random stain perturbation.
        :param dropout: Dropout rate for regularization.
        """
        super(SAStainDiff, self).__init__()
        
        if stain_emb_dim <= 0:
            raise ValueError("stain_emb_dim must be strictly positive.")
            
        self.stain_emb_dim = stain_emb_dim
        self.perturbation_dim = in_channels * 2
        
        # 1. Stain Augmenter
        self.augmenter = StainAugmenter(perturbation_scale=perturbation_scale)
        
        # 2. Self-Supervised Stain Encoder
        self.stain_encoder = SelfSupervisedStainEncoder(
            in_channels=in_channels, 
            embed_dim=stain_emb_dim, 
            perturbation_dim=self.perturbation_dim
        )
        
        # 3. Time Embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, stain_emb_dim) # Project to match stain_emb_dim for additive fusion
        )
        
        # 4. UNet Backbone
        self.unet = SAStainDiffUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            cond_dim=stain_emb_dim,
            dropout=dropout
        )

    def compute_self_supervised_loss(self, delta_pred: torch.Tensor, delta_target: torch.Tensor) -> torch.Tensor:
        """
        Computes the self-supervised loss for predicting the applied stain perturbation.
        
        Equation: L_SS = || delta_pred - delta_target ||_2^2
        
        :param delta_pred: Predicted perturbation vector (B, perturbation_dim).
        :param delta_target: Ground truth perturbation vector (B, perturbation_dim).
        :return: Scalar self-supervised loss.
        """
        return F.mse_loss(delta_pred, delta_target)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x_orig: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SAStainDiff baseline model.
        
        Mathematical Formulation:
        1. x_aug, delta = StainAugmenter(x_orig)
        2. v_stain, delta_pred = Encoder(x_aug)
        3. z_t = MLP_time(t)
        4. z_final = z_t + v_stain (Additive Fusion)
        5. epsilon_pred = UNet(x_t, z_final)
        
        :param x_t: Noisy input image of shape (B, C_in, H, W).
        :param t: Diffusion timestep of shape (B,).
        :param x_orig: Original clean image for self-supervised augmentation (B, C_in, H, W).
                       If None, x_t is used as the base for augmentation (for inference).
        :return: Tuple containing predicted noise epsilon_pred (B, C_out, H, W) 
                 and predicted perturbation delta_pred (B, perturbation_dim).
        """
        if x_t.dim() != 4:
            raise ValueError(f"Expected input image x_t to be 4D (B, C, H, W), got {x_t.dim()}D.")
        if t.dim() != 1:
            raise ValueError(f"Expected timestep t to be 1D (B,), got {t.dim()}D.")
            
        # 1. Stain Augmentation
        base_img = x_orig if x_orig is not None else x_t.detach()
        x_aug, delta_target = self.augmenter(base_img)
        
        # 2. Extract Stain Embedding and Predict Perturbation
        v_stain, delta_pred = self.stain_encoder(x_aug)
        
        # 3. Compute Time Embedding
        z_t = self.time_mlp(t.float())
        
        # 4. Additive Fusion (The core flaw causing entanglement in baselines)
        z_final = z_t + v_stain
        
        # 5. Predict Noise via UNet
        epsilon_pred = self.unet(x_t, z_final)
        
        return epsilon_pred, delta_pred, delta_target


if __name__ == "__main__":
    # Example usage and validation of the SAStainDiff baseline model
    
    batch_size = 4
    in_channels = 3
    out_channels = 3
    height, width = 64, 64
    base_channels = 32
    channel_multipliers = [1, 2, 4]
    time_emb_dim = 128
    stain_emb_dim = 128
    
    model = SAStainDiff(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        time_emb_dim=time_emb_dim,
        stain_emb_dim=stain_emb_dim,
        perturbation_scale=0.2,
        dropout=0.1
    )
    
    print(f"Initialized SAStainDiff Baseline:")
    print(f"  Stain Embedding Dimension (d_stain): {stain_emb_dim}")
    print(f"  Perturbation Dimension: {model.perturbation_dim}")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x_t_orig = torch.randn(batch_size, in_channels, height, width)
    x_orig_orig = torch.rand(batch_size, in_channels, height, width)
    t_orig = torch.randint(0, 1000, (batch_size,))
    
    epsilon_pred, delta_pred, delta_target = model(x_t_orig, t_orig, x_orig_orig)
    
    print(f"\nForward Pass Validation:")
    print(f"  Input x_t shape: {x_t_orig.shape}")
    print(f"  Output epsilon_pred shape: {epsilon_pred.shape}")
    print(f"  Predicted delta shape: {delta_pred.shape}")
    print(f"  Target delta shape: {delta_target.shape}")
    
    assert epsilon_pred.shape == (batch_size, out_channels, height, width), "Output shape mismatch!"
    assert delta_pred.shape == (batch_size, model.perturbation_dim), "Delta pred shape mismatch!"
    
    # Compute losses
    L_diff = F.mse_loss(epsilon_pred, torch.randn_like(epsilon_pred))
    L_ss = model.compute_self_supervised_loss(delta_pred, delta_target)
    L_total = L_diff + 0.1 * L_ss
    
    L_total.backward()
    
    assert model.stain_encoder.encoder[0].weight.grad is not None, "Gradients did not flow to stain encoder!"
    assert model.unet.init_conv.weight.grad is not None, "Gradients did not flow to UNet!"
    
    print("\nVerification passed: SAStainDiff shapes are correct and gradients flow successfully through the self-supervised and diffusion pipelines.")