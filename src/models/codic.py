"""
codic.py
Implements the CoDiC (Contrastive Disentanglement) baseline model.
CoDiC enforces contrastive disentanglement to separate class (biological) 
embeddings from instance-specific (physical/style) embeddings using 
supervised contrastive learning. 

Unlike PhyBio-ODM, CoDiC relies on contrastive alignment in a shared 
Euclidean space without explicit continuous physical parameterization 
or orthogonal manifold constraints, failing to capture the non-linear 
physical coupling of tissue and chemical dyes.

Mathematical Formulations:
1. Dual Encoding: 
   v_class = E_class(x), v_inst = E_inst(x)
2. Contrastive Disentanglement Loss (InfoNCE-style):
   L_contrast = - 1/|P(i)| sum_{p in P(i)} log [ exp(v_class_i . v_class_p / tau) / sum_{k != i} exp(v_class_i . v_inst_k / tau) ]
3. Additive Conditioning: 
   z_cond = z_t + v_class + v_inst
4. Noise Prediction: 
   epsilon_pred = UNet(x_t, z_cond)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# Reuse components from the unet_backbone
from .unet_backbone import SinusoidalPositionalEmbedding, Downsample, Upsample


class ClassEncoder(nn.Module):
    """
    Extracts class (biological/morphological) embeddings from the input image.
    
    Mathematical Equation:
    v_class = E_class(x)
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 256):
        super(ClassEncoder, self).__init__()
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
        :return: Class embedding v_class of shape (B, embed_dim).
        """
        return self.encoder(x)


class InstanceEncoder(nn.Module):
    """
    Extracts instance (physical/style/stain) embeddings from the input image.
    
    Mathematical Equation:
    v_inst = E_inst(x)
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 256):
        super(InstanceEncoder, self).__init__()
        # Uses a separate branch to simulate instance/style extraction
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
        :return: Instance embedding v_inst of shape (B, embed_dim).
        """
        return self.encoder(x)


class ContrastiveDisentanglementLoss(nn.Module):
    """
    Computes the contrastive disentanglement loss to separate class and instance embeddings.
    Pulls together class embeddings of the same class while pushing apart class and instance embeddings.
    
    Mathematical Equation:
    L_contrast = - 1/|P(i)| sum_{p in P(i)} log [ exp(v_class_i . v_class_p / tau) / sum_{k != i} exp(v_class_i . v_inst_k / tau) ]
    """
    def __init__(self, temperature: float = 0.1):
        super(ContrastiveDisentanglementLoss, self).__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be strictly positive.")
        self.temperature = temperature

    def forward(self, v_class: torch.Tensor, v_inst: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        :param v_class: Class embeddings of shape (B, d).
        :param v_inst: Instance embeddings of shape (B, d).
        :param labels: Class labels of shape (B,) used to identify positive pairs.
        :return: Scalar contrastive loss.
        """
        # Normalize embeddings to compute cosine similarity
        v_class_norm = F.normalize(v_class, dim=1)
        v_inst_norm = F.normalize(v_inst, dim=1)
        
        # Compute similarity matrices
        # sim_class: (B, B) similarity between class embeddings
        sim_class = torch.matmul(v_class_norm, v_class_norm.T) / self.temperature
        
        # sim_cross: (B, B) similarity between class and instance embeddings (denominator)
        sim_cross = torch.matmul(v_class_norm, v_inst_norm.T) / self.temperature
        
        # Create mask for positive pairs (same class, excluding self)
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(v_class.device)
        mask_self = torch.eye(mask_pos.shape[0], device=v_class.device)
        mask_pos = mask_pos - mask_self  # Exclude self
        
        # Compute number of positive pairs for each anchor
        pos_counts = mask_pos.sum(dim=1)
        pos_counts = torch.clamp(pos_counts, min=1.0)  # Avoid division by zero
        
        # Compute log-softmax for the denominator (cross similarity)
        # We use the cross similarity as the negative samples
        logits_max, _ = torch.max(sim_cross, dim=1, keepdim=True)
        logits = sim_cross - logits_max.detach()
        exp_logits = torch.exp(logits)
        
        # Sum of exp(logits) for the denominator
        # We exclude the diagonal (self vs self instance) if needed, but here we just sum all
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean log-likelihood over positive pairs
        # We use the class-class similarity as the numerator
        # Actually, standard SupCon uses class-class for numerator and all others for denominator.
        # Here we adapt it to separate class from instance:
        # Numerator: exp(sim_class) for positive pairs
        # Denominator: sum(exp(sim_cross))
        
        # Recompute log_prob specifically for our contrastive objective:
        # We want to maximize sim_class for positive pairs, and minimize sim_cross for all pairs.
        # Let's use a simplified InfoNCE where positive pairs are from sim_class, and negatives are from sim_cross.
        
        exp_sim_class_pos = torch.exp(sim_class) * mask_pos
        sum_exp_sim_cross = exp_logits.sum(dim=1, keepdim=True)
        
        # Probability of positive pair
        prob = exp_sim_class_pos / (sum_exp_sim_cross + 1e-8)
        prob = torch.clamp(prob, min=1e-8)  # Avoid log(0)
        
        # Loss
        loss = - (torch.log(prob) * mask_pos).sum(dim=1) / pos_counts
        loss = loss.mean()
        
        return loss


class CoDiCResidualBlock(nn.Module):
    """
    Residual block for CoDiC, conditioned on the additively fused class and instance embeddings.
    Lacks the orthogonal dual-stream modulation of PhyBio-ODM.
    """
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.0):
        super(CoDiCResidualBlock, self).__init__()
        
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


class CoDiCUNet(nn.Module):
    """
    UNet backbone for CoDiC, utilizing the additively fused class and instance conditioning.
    """
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3, 
                 base_channels: int = 64, 
                 channel_multipliers: List[int] = [1, 2, 4], 
                 cond_dim: int = 256,
                 dropout: float = 0.1):
        super(CoDiCUNet, self).__init__()
        
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList()
        channels = [base_channels]
        current_channels = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            self.downs.append(nn.ModuleList([
                CoDiCResidualBlock(current_channels, out_ch, cond_dim, dropout),
                CoDiCResidualBlock(out_ch, out_ch, cond_dim, dropout)
            ]))
            channels.extend([out_ch, out_ch])
            current_channels = out_ch
            
            if i != len(channel_multipliers) - 1:
                self.downs.append(nn.ModuleList([Downsample(current_channels), None]))
                channels.append(current_channels)
                
        mid_channels = base_channels * channel_multipliers[-1]
        self.mid_block1 = CoDiCResidualBlock(mid_channels, mid_channels, cond_dim, dropout)
        self.mid_block2 = CoDiCResidualBlock(mid_channels, mid_channels, cond_dim, dropout)
        
        self.ups = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            in_ch = current_channels + channels.pop()
            
            self.ups.append(nn.ModuleList([
                CoDiCResidualBlock(in_ch, out_ch, cond_dim, dropout),
                CoDiCResidualBlock(out_ch, out_ch, cond_dim, dropout)
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


class CoDiC(nn.Module):
    """
    Full CoDiC (Contrastive Disentanglement) baseline model.
    Integrates dual-branch encoding with contrastive disentanglement loss.
    """
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 base_channels: int = 64,
                 channel_multipliers: List[int] = [1, 2, 4],
                 time_emb_dim: int = 256,
                 embed_dim: int = 256,
                 temperature: float = 0.1,
                 dropout: float = 0.1):
        super(CoDiC, self).__init__()
        
        if embed_dim <= 0:
            raise ValueError("embed_dim must be strictly positive.")
            
        self.embed_dim = embed_dim
        
        # 1. Dual Encoders for Class and Instance
        self.class_encoder = ClassEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.instance_encoder = InstanceEncoder(in_channels=in_channels, embed_dim=embed_dim)
        
        # 2. Contrastive Disentanglement Loss
        self.contrast_loss = ContrastiveDisentanglementLoss(temperature=temperature)
        
        # 3. Time Embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, embed_dim)
        )
        
        # 4. UNet Backbone
        self.unet = CoDiCUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            cond_dim=embed_dim,
            dropout=dropout
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x_orig: torch.Tensor = None, labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the CoDiC baseline model.
        
        Mathematical Formulation:
        1. v_class = E_class(x_orig), v_inst = E_inst(x_orig)
        2. z_t = MLP_time(t)
        3. z_cond = z_t + v_class + v_inst (Additive Fusion)
        4. epsilon_pred = UNet(x_t, z_cond)
        
        :param x_t: Noisy input image of shape (B, C_in, H, W).
        :param t: Diffusion timestep of shape (B,).
        :param x_orig: Original clean image for dual encoding (B, C_in, H, W).
                       If None, x_t is used as the base for encoding (for inference).
        :param labels: Class labels of shape (B,) for contrastive loss.
        :return: Tuple containing predicted noise epsilon_pred (B, C_out, H, W), 
                 class embedding v_class (B, d), instance embedding v_inst (B, d),
                 and contrastive loss L_contrast (scalar).
        """
        if x_t.dim() != 4:
            raise ValueError(f"Expected input image x_t to be 4D (B, C, H, W), got {x_t.dim()}D.")
        if t.dim() != 1:
            raise ValueError(f"Expected timestep t to be 1D (B,), got {t.dim()}D.")
            
        # 1. Dual Encoding
        base_img = x_orig if x_orig is not None else x_t.detach()
        v_class = self.class_encoder(base_img)
        v_inst = self.instance_encoder(base_img)
        
        # 2. Compute Time Embedding
        z_t = self.time_mlp(t.float())
        
        # 3. Additive Fusion (The core flaw causing entanglement in baselines)
        z_cond = z_t + v_class + v_inst
        
        # 4. Predict Noise via UNet
        epsilon_pred = self.unet(x_t, z_cond)
        
        # 5. Compute Contrastive Loss (if labels are provided during training)
        if labels is not None:
            l_contrast = self.contrast_loss(v_class, v_inst, labels)
        else:
            l_contrast = torch.tensor(0.0, device=x_t.device)
        
        return epsilon_pred, v_class, v_inst, l_contrast


if __name__ == "__main__":
    # Example usage and validation of the CoDiC baseline model
    
    batch_size = 4
    in_channels = 3
    out_channels = 3
    height, width = 64, 64
    base_channels = 32
    channel_multipliers = [1, 2, 4]
    time_emb_dim = 128
    embed_dim = 128
    num_classes = 10
    
    model = CoDiC(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        time_emb_dim=time_emb_dim,
        embed_dim=embed_dim,
        temperature=0.1,
        dropout=0.1
    )
    
    print(f"Initialized CoDiC Baseline:")
    print(f"  Embedding Dimension (d): {embed_dim}")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x_t_orig = torch.randn(batch_size, in_channels, height, width)
    x_orig_orig = torch.rand(batch_size, in_channels, height, width)
    t_orig = torch.randint(0, 1000, (batch_size,))
    labels_orig = torch.randint(0, num_classes, (batch_size,))
    
    epsilon_pred, v_class, v_inst, l_contrast = model(x_t_orig, t_orig, x_orig_orig, labels_orig)
    
    print(f"\nForward Pass Validation:")
    print(f"  Input x_t shape: {x_t_orig.shape}")
    print(f"  Output epsilon_pred shape: {epsilon_pred.shape}")
    print(f"  Class embedding v_class shape: {v_class.shape}")
    print(f"  Instance embedding v_inst shape: {v_inst.shape}")
    print(f"  Contrastive loss: {l_contrast.item():.4f}")
    
    assert epsilon_pred.shape == (batch_size, out_channels, height, width), "Output shape mismatch!"
    assert v_class.shape == (batch_size, embed_dim), "v_class shape mismatch!"
    assert v_inst.shape == (batch_size, embed_dim), "v_inst shape mismatch!"
    
    # Compute total loss
    L_diff = F.mse_loss(epsilon_pred, torch.randn_like(epsilon_pred))
    L_total = L_diff + 0.1 * l_contrast
    
    L_total.backward()
    
    assert model.class_encoder.encoder[0].weight.grad is not None, "Gradients did not flow to class encoder!"
    assert model.instance_encoder.encoder[0].weight.grad is not None, "Gradients did not flow to instance encoder!"
    assert model.unet.init_conv.weight.grad is not None, "Gradients did not flow to UNet!"
    
    print("\nVerification passed: CoDiC shapes are correct and gradients flow successfully through the dual-encoder and contrastive pipelines.")