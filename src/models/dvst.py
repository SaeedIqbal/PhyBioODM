"""
dvst.py
Implements the D-VST (Diffusion Virtual Staining Transformer) baseline model.
D-VST introduces dual-encoder Diffusion Transformers to separate pathology 
and tone information to address "pathology leakage". Unlike PhyBio-ODM, 
it relies on dual-encoder feature separation without explicit continuous 
physical parameterization or orthogonal manifold constraints.

Mathematical Formulations:
1. Dual Encoding: 
   v_path = E_path(x), v_tone = E_tone(x)
2. Patch Embedding: 
   z = Conv2d(x_t) + PE
3. DiT Block Conditioning (AdaLN):
   h = gamma(v) * LayerNorm(x) + beta(v)
4. Diffusion Process:
   epsilon_pred = DiT(x_t, t, v_path, v_tone)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# Reuse components from the proposed methodology
from .unet_backbone import SinusoidalPositionalEmbedding


class PatchEmbed2d(nn.Module):
    """
    Converts image patches to token embeddings for the Transformer.
    
    Mathematical Equation:
    z = Conv2d(x_t) + PE
    """
    def __init__(self, img_size: int = 64, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input image tensor of shape (B, C, H, W).
        :return: Patch tokens of shape (B, num_patches, embed_dim).
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds the diffusion timestep using sinusoidal embeddings and an MLP.
    """
    def __init__(self, time_emb_dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        self.sinusoidal = SinusoidalPositionalEmbedding(time_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, time_emb_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        :param t: Diffusion timestep of shape (B,).
        :return: Timestep embedding of shape (B, time_emb_dim).
        """
        t_emb = self.sinusoidal(t)
        return self.mlp(t_emb)


class DualEncoder(nn.Module):
    """
    Dual encoder to separate pathology (content) and tone (style) information.
    
    Mathematical Equations:
    v_path = E_path(x)
    v_tone = E_tone(x)
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        
        # Pathology Encoder (Focuses on structural/morphological features)
        self.path_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )
        
        # Tone Encoder (Focuses on color/stain features)
        # Uses a separate branch to simulate tone extraction without physical grounding
        self.tone_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: Input image tensor of shape (B, C, H, W).
        :return: Tuple containing v_path (B, embed_dim) and v_tone (B, embed_dim).
        """
        v_path = self.path_encoder(x)
        v_tone = self.tone_encoder(x)
        return v_path, v_tone


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with AdaLN conditioning.
    
    Mathematical Equation:
    h = gamma(v) * LayerNorm(x) + beta(v)
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, cond_dim: int = 256):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size)
        )
        
        # AdaLN modulation for dual conditioning (pathology and tone)
        # We concatenate v_path and v_tone for modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim * 2, hidden_size * 6)  # gamma1, beta1, alpha1, gamma2, beta2, alpha2
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        :param x: Token sequence of shape (B, N, hidden_size).
        :param c: Conditioning vector [v_path || v_tone] of shape (B, 2 * cond_dim).
        :return: Modulated token sequence of shape (B, N, hidden_size).
        """
        mod = self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod
        
        # Attention block
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP block
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x


class FinalLayer(nn.Module):
    """
    Final layer of DiT to unpatchify the tokens back to image space.
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, cond_dim: int = 256):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim * 2, hidden_size * 2)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        :param x: Token sequence of shape (B, N, hidden_size).
        :param c: Conditioning vector [v_path || v_tone] of shape (B, 2 * cond_dim).
        :return: Unpatchified tokens of shape (B, N, patch_size * patch_size * out_channels).
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class DVST(nn.Module):
    """
    Full D-VST (Diffusion Virtual Staining Transformer) baseline model.
    """
    def __init__(self,
                 img_size: int = 64,
                 patch_size: int = 4,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 hidden_size: int = 256,
                 depth: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 cond_dim: int = 256):
        super().__init__()
        
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size.")
            
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1. Dual Encoders for Pathology and Tone
        self.dual_encoder = DualEncoder(in_channels=in_channels, embed_dim=cond_dim)
        
        # 2. Patch Embedding
        self.patch_embed = PatchEmbed2d(img_size, patch_size, in_channels, hidden_size)
        
        # 3. Timestep Embedder (Output dimension matches cond_dim for addition)
        self.time_embed = TimestepEmbedder(time_emb_dim=cond_dim, hidden_dim=cond_dim * 4)
        
        # 4. DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, cond_dim) for _ in range(depth)
        ])
        
        # 5. Final Layer
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels, cond_dim)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the modulation layers to zero to ensure the model 
        behaves as an identity function at the start of training.
        """
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts token sequence back to image tensor.
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the D-VST baseline model.
        
        Mathematical Formulation:
        1. v_path, v_tone = DualEncoder(x_t)
        2. z = PatchEmbed(x_t)
        3. c_time = TimeEmbed(t)
        4. c_cond = [ (v_path + c_time) || (v_tone + c_time) ]
        5. For each block: z = DiTBlock(z, c_cond)
        6. x_out = Unpatchify(FinalLayer(z, c_cond))
        
        :param x_t: Noisy input image of shape (B, C, H, W).
        :param t: Diffusion timestep of shape (B,).
        :return: Predicted noise epsilon_pred of shape (B, C, H, W).
        """
        if x_t.dim() != 4:
            raise ValueError(f"Expected x_t to be 4D, got {x_t.dim()}D")
        if t.dim() != 1:
            raise ValueError(f"Expected t to be 1D, got {t.dim()}D")
            
        # 1. Dual Encoding
        v_path, v_tone = self.dual_encoder(x_t)
        
        # 2. Timestep Embedding
        c_time = self.time_embed(t)  # (B, cond_dim)
        
        # Add time embedding to both pathology and tone conditions
        v_path_cond = v_path + c_time
        v_tone_cond = v_tone + c_time
        
        # Concatenate for dual conditioning: c_cond = [v_path_cond || v_tone_cond]
        c_cond = torch.cat([v_path_cond, v_tone_cond], dim=-1)  # (B, 2 * cond_dim)
        
        # 3. Patch Embedding
        x = self.patch_embed(x_t)
        
        # 4. DiT Blocks
        for block in self.blocks:
            x = block(x, c_cond)
            
        # 5. Final Layer and Unpatchify
        x = self.final_layer(x, c_cond)
        x = self.unpatchify(x)
        
        return x


if __name__ == "__main__":
    # Example usage and validation of the D-VST baseline model
    
    batch_size = 4
    img_size = 64
    in_channels = 3
    out_channels = 3
    
    model = DVST(
        img_size=img_size,
        patch_size=4,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=256,
        depth=4,
        num_heads=8,
        cond_dim=256
    )
    
    print(f"Initialized D-VST Baseline:")
    print(f"  Image Size: {img_size}")
    print(f"  Patch Size: {model.patch_size}")
    print(f"  Hidden Size: 256")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x_t_orig = torch.randn(batch_size, in_channels, img_size, img_size)
    t_orig = torch.randint(0, 1000, (batch_size,))
    
    epsilon_pred = model(x_t_orig, t_orig)
    
    print(f"\nForward Pass Validation:")
    print(f"  Input x_t shape: {x_t_orig.shape}")
    print(f"  Output epsilon_pred shape: {epsilon_pred.shape}")
    
    assert epsilon_pred.shape == (batch_size, out_channels, img_size, img_size), "Output shape mismatch!"
    
    loss = epsilon_pred.sum()
    loss.backward()
    
    assert model.dual_encoder.path_encoder[0].weight.grad is not None, "Gradients did not flow to path encoder!"
    assert model.patch_embed.proj.weight.grad is not None, "Gradients did not flow to patch embed!"
    
    print("\nVerification passed: D-VST shapes are correct and gradients flow successfully through the dual-encoder Transformer pipeline.")