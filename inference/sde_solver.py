"""
sde_solver.py
Numerical solver for the reverse-time Stochastic Differential Equation (SDE).
Guides the generation process from pure noise to clean histopathology patches 
conditioned on the continuous physical embeddings of unseen domains.

Mathematical Formulation:
Reverse SDE: dx_t = [f(x_t, t) - g(t)^2 \nabla log p_theta(x_t | t, v_bio, v_phys_new)] dt + g(t) dw_t
where:
  f(x_t, t) = -0.5 * beta(t) * x_t  (Drift coefficient)
  g(t) = sqrt(beta(t))               (Diffusion coefficient)
  \nabla log p_theta \approx -epsilon_theta / sqrt(1 - alpha_bar_t) (Score function approximation)

Euler-Maruyama Discretization (Backward in time):
x_{t-1} = x_t - [f(x_t, t) - g(t)^2 \nabla log p_theta] * dt + g(t) * sqrt(dt) * z
where z ~ N(0, I)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

class ReverseSDESolver:
    """
    Implements the Euler-Maruyama numerical solver for the reverse-time SDE 
    used in the PhyBio-ODM framework.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 num_steps: int = 1000, 
                 beta_start: float = 1e-4, 
                 beta_end: float = 0.02, 
                 device: str = 'cuda'):
        """
        :param model: The trained PhyBio-ODM UNet backbone.
        :param num_steps: Number of discretization steps (T).
        :param beta_start: Starting variance of the noise schedule.
        :param beta_end: Ending variance of the noise schedule.
        :param device: Computation device.
        """
        self.model = model
        self.num_steps = num_steps
        self.device = device
        
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Precompute discrete alpha_bars for score function approximation
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _get_continuous_beta(self, t_continuous: torch.Tensor) -> torch.Tensor:
        """
        Computes the continuous variance schedule beta(t).
        Equation: beta(t) = beta_start + t * (beta_end - beta_start)
        """
        return self.beta_start + t_continuous * (self.beta_end - self.beta_start)

    def _compute_score_function(self, 
                                x_t: torch.Tensor, 
                                t_continuous: torch.Tensor, 
                                v_bio: torch.Tensor, 
                                v_phys_new: torch.Tensor) -> torch.Tensor:
        """
        Approximates the score function \nabla_{x_t} log p_theta using the network's noise prediction.
        Equation: \nabla log p_theta \approx -epsilon_theta(x_t, t, c) / sqrt(1 - alpha_bar_t)
        """
        # Map continuous time t in [0, 1] to discrete index [0, T-1]
        t_discrete = (t_continuous * (self.num_steps - 1)).long().clamp(0, self.num_steps - 1)
        
        # Query the model for noise prediction
        # Assuming the model has a method to accept latent embeddings directly for inference
        epsilon_pred = self.model.predict_noise_with_latents(x_t, t_discrete, v_bio, v_phys_new)
        
        # Retrieve alpha_bar for the current timestep
        alpha_bar_t = self.alpha_bars[t_discrete].view(-1, 1, 1, 1)
        
        # Compute score approximation
        score = -epsilon_pred / torch.sqrt(1.0 - alpha_bar_t + 1e-8)
        
        return score

    def euler_maruyama_step(self, 
                            x_t: torch.Tensor, 
                            t_continuous: torch.Tensor, 
                            dt: float, 
                            v_bio: torch.Tensor, 
                            v_phys_new: torch.Tensor) -> torch.Tensor:
        """
        Performs a single backward Euler-Maruyama discretization step.
        Equation: x_{t-1} = x_t - [f(x_t, t) - g(t)^2 \nabla log p_theta] * dt + g(t) * sqrt(dt) * z
        """
        # 1. Compute continuous beta(t)
        beta_t = self._get_continuous_beta(t_continuous).view(-1, 1, 1, 1)
        
        # 2. Compute drift coefficient f(x_t, t) = -0.5 * beta(t) * x_t
        f_xt = -0.5 * beta_t * x_t
        
        # 3. Compute diffusion coefficient g(t) = sqrt(beta(t))
        g_t = torch.sqrt(beta_t)
        
        # 4. Compute score function \nabla log p_theta
        score = self._compute_score_function(x_t, t_continuous, v_bio, v_phys_new)
        
        # 5. Compute deterministic drift term: f(x_t, t) - g(t)^2 * score
        drift = f_xt - (g_t ** 2) * score
        
        # 6. Compute stochastic diffusion term: g(t) * sqrt(dt) * z
        z = torch.randn_like(x_t)
        diffusion = g_t * np.sqrt(dt) * z
        
        # 7. Update state backward in time
        x_prev = x_t - drift * dt + diffusion
        
        return x_prev

    @torch.no_grad()
    def solve(self, x_T: torch.Tensor, v_bio: torch.Tensor, v_phys_new: torch.Tensor) -> torch.Tensor:
        """
        Solves the reverse SDE from t=1 (noise) to t=0 (clean image).
        
        :param x_T: Initial pure Gaussian noise of shape (B, C, H, W).
        :param v_bio: Biological latent embedding of shape (B, d_bio).
        :param v_phys_new: Physical latent embedding of the unseen domain of shape (B, d_phys).
        :return: Generated clean images x_0 of shape (B, C, H, W).
        """
        self.model.eval()
        x_t = x_T.clone()
        
        # Time step size dt = 1 / T
        dt = 1.0 / self.num_steps
        
        # Loop backward from t=1.0 to t=0.0
        t_values = torch.linspace(1.0, 0.0, self.num_steps + 1)[:-1].to(self.device)
        
        for t_cont in t_values:
            # Expand t_cont to match batch dimension
            t_batch = torch.full((x_t.shape[0],), t_cont.item(), device=self.device)
            
            # Perform one Euler-Maruyama step
            x_t = self.euler_maruyama_step(x_t, t_batch, dt, v_bio, v_phys_new)
            
        # Final clamp to ensure valid image range if necessary (optional, depends on data normalization)
        x_0 = x_t
        
        return x_0


if __name__ == "__main__":
    # Mock model for demonstration
    class MockUNet(nn.Module):
        def __init__(self):
            super().__init__()
        def predict_noise_with_latents(self, x_t, t, v_bio, v_phys):
            # Simulate a simple noise prediction
            return torch.randn_like(x_t) * 0.5

    # Initialize solver
    model = MockUNet()
    solver = ReverseSDESolver(model, num_steps=100, device='cpu')
    
    # orig inputs
    batch_size = 2
    x_T = torch.randn(batch_size, 3, 64, 64, device='cpu')
    v_bio = torch.randn(batch_size, 256, device='cpu')
    v_phys_new = torch.randn(batch_size, 256, device='cpu')
    
    # Solve reverse SDE
    x_0 = solver.solve(x_T, v_bio, v_phys_new)
    
    print(f"Initial noise x_T shape: {x_T.shape}")
    print(f"Generated clean image x_0 shape: {x_0.shape}")
    print("Reverse SDE solver executed successfully.")