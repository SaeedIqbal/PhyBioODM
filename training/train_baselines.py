"""
train_baselines.py
Unified training script for all 7 SOTA baseline models:
MeDi, CytoSyn, CHIS, SAStainDiff, D-VST, FedSD, and CoDiC.

Implements specific loss formulations for each baseline to ensure fair comparison.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

class BaselineTrainer:
    """
    Trainer class for SOTA baseline models.
    Handles the distinct conditioning mechanisms and loss functions of each baseline.
    """
    
    def __init__(self, 
                 model_type: str,
                 model: nn.Module, 
                 dataloaders: Dict[str, DataLoader], 
                 optimizer: torch.optim.Optimizer,
                 num_timesteps: int = 1000,
                 device: str = 'cuda'):
        """
        :param model_type: String identifier for the baseline ('MeDi', 'CytoSyn', 'CHIS', 'SAStainDiff', 'D-VST', 'FedSD', 'CoDiC').
        :param model: The baseline neural network.
        :param dataloaders: Dictionary containing 'train' and 'val' DataLoaders.
        :param optimizer: PyTorch optimizer.
        :param num_timesteps: Total number of diffusion steps (T).
        :param device: Device to train on.
        """
        if model_type not in ['MeDi', 'CytoSyn', 'CHIS', 'SAStainDiff', 'D-VST', 'FedSD', 'CoDiC']:
            raise ValueError(f"Unsupported baseline model type: {model_type}")
            
        self.model_type = model_type
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Precompute noise schedule
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Baseline-specific hyperparameters
        self.lambda_contrast = 0.1 if model_type == 'CoDiC' else 0.0
        self.lambda_ortho_abstract = 0.1 if model_type == 'FedSD' else 0.0
        self.lambda_ss = 0.1 if model_type == 'SAStainDiff' else 0.0

    def forward_diffusion(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard forward diffusion process."""
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        return x_t, noise

    def compute_loss_medi(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor, metadata: Dict[str, torch.Tensor], noise: torch.Tensor) -> torch.Tensor:
        """
        MeDi Loss: Standard diffusion loss with discrete additive conditioning.
        L_diff = || epsilon - epsilon_theta(x_t, t, y, metadata) ||_2^2
        """
        epsilon_pred = self.model(x_t, t, y, metadata)
        return nn.functional.mse_loss(epsilon_pred, noise)

    def compute_loss_codic(self, x_t: torch.Tensor, t: torch.Tensor, x_orig: torch.Tensor, y: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        CoDiC Loss: Diffusion loss + Contrastive Disentanglement Loss.
        L_total = L_diff + lambda_contrast * L_contrast
        """
        epsilon_pred, v_class, v_inst, l_contrast = self.model(x_t, t, x_orig, y)
        l_diff = nn.functional.mse_loss(epsilon_pred, noise)
        return l_diff + self.lambda_contrast * l_contrast

    def compute_loss_fedsd(self, x_t: torch.Tensor, t: torch.Tensor, x_orig: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        FedSD Loss: Diffusion loss + Abstract Orthogonality Loss.
        L_total = L_diff + lambda_ortho * || v_sem^T v_sty ||_F^2
        """
        epsilon_pred, v_sem, v_sty = self.model(x_t, t, x_orig)
        l_diff = nn.functional.mse_loss(epsilon_pred, noise)
        
        # Abstract orthogonality (dot product minimization)
        l_ortho = torch.norm(v_sem.T @ v_sty, p='fro') ** 2
        return l_diff + self.lambda_ortho_abstract * l_ortho

    def compute_loss_sastaindiff(self, x_t: torch.Tensor, t: torch.Tensor, x_orig: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        SAStainDiff Loss: Diffusion loss + Self-Supervised Stain Prediction Loss.
        L_total = L_diff + lambda_ss * || delta_pred - delta_target ||_2^2
        """
        epsilon_pred, delta_pred, delta_target = self.model(x_t, t, x_orig)
        l_diff = nn.functional.mse_loss(epsilon_pred, noise)
        l_ss = nn.functional.mse_loss(delta_pred, delta_target)
        return l_diff + self.lambda_ss * l_ss

    def compute_loss_generic(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Generic loss for CytoSyn, CHIS, D-VST which primarily rely on standard diffusion loss 
        with their respective coupled/frequency/dual-encoder conditioning.
        """
        # Assuming these models return just the noise prediction in their forward pass
        epsilon_pred = self.model(x_t, t, **kwargs)
        return nn.functional.mse_loss(epsilon_pred, noise)

    def train_step(self, batch: Tuple) -> Dict[str, float]:
        """
        Performs a single training step, routing to the appropriate loss function based on model_type.
        """
        self.optimizer.zero_grad()
        
        # Unpack batch based on model requirements
        x_0 = batch[0].to(self.device)
        B = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        x_t, noise = self.forward_diffusion(x_0, t)
        
        # Route to specific loss computation
        if self.model_type == 'MeDi':
            y, metadata = batch[1].to(self.device), {k: v.to(self.device) for k, v in batch[2].items()}
            loss = self.compute_loss_medi(x_t, t, y, metadata, noise)
            
        elif self.model_type == 'CoDiC':
            y = batch[1].to(self.device)
            loss = self.compute_loss_codic(x_t, t, x_0, y, noise)
            
        elif self.model_type == 'FedSD':
            loss = self.compute_loss_fedsd(x_t, t, x_0, noise)
            
        elif self.model_type == 'SAStainDiff':
            loss = self.compute_loss_sastaindiff(x_t, t, x_0, noise)
            
        else: # CytoSyn, CHIS, D-VST
            # Mocking additional inputs for these baselines
            kwargs = {'metadata': {}, 'm_prior': x_0} 
            loss = self.compute_loss_generic(x_t, t, noise, **kwargs)
            
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}

    def train(self, num_epochs: int):
        """Main training loop for baselines."""
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            
            progress_bar = tqdm(self.dataloaders['train'], desc=f"[{self.model_type}] Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(progress_bar):
                losses = self.train_step(batch)
                total_loss += losses['loss']
                progress_bar.set_postfix({'loss': f"{total_loss/(batch_idx+1):.4f}"})
                
            print(f"Epoch {epoch+1} Average Loss: {total_loss / len(self.dataloaders['train']):.4f}")