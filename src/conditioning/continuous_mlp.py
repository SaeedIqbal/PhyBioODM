"""
continuous_mlp.py
Implements the Physics-aware Multi-Layer Perceptron (MLP) that maps the continuous 
physical parameter vector p to the latent physical embedding v_phys.

This module resolves Gap 2 (Additive Morphological Entanglement) by replacing 
discrete token embeddings with a continuous, non-linear mapping grounded in 
tissue optics, enabling the orthogonal dual-stream architecture to modulate 
the UNet without distorting biological geometry.

Mathematical Formulation:
v_phys = phi_phys(p) = W_L * sigma( W_{L-1} ... sigma(W_1 * p + b_1) ... + b_{L-1} ) + b_L

Where:
- p in R^d is the continuous physical parameter vector (from parameter_aggregation.py).
- W_l, b_l are learnable weights and biases.
- sigma is the Swish activation function: sigma(x) = x * sigmoid(x).
- v_phys in R^{d_phys} is the output latent physical embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class SwishActivation(nn.Module):
    """
    Implements the Swish activation function (also known as SiLU).
    
    Mathematical Equation:
    sigma(x) = x * sigmoid(x) = x / (1 + exp(-x))
    
    Swish is used in the proposed methodology to provide smooth, non-monotonic 
    non-linearities that help the network learn the complex, continuous physical 
    manifolds of stain chemistry and scanner optics.
    """
    
    def __init__(self):
        super(SwishActivation, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the Swish activation function element-wise.
        
        :param x: Input tensor of any shape.
        :return: Activated tensor of the same shape.
        """
        return x * torch.sigmoid(x)


class PhysicsAwareMLP(nn.Module):
    """
    Physics-aware Multi-Layer Perceptron (phi_phys) that projects the continuous 
    physical parameter vector p into the latent conditioning space v_phys.
    
    By using continuous physical parameters instead of discrete categorical tokens, 
    this MLP allows the model to interpolate smoothly across the physical parameter 
    space, enabling true zero-shot generalization to unseen medical centers.
    """
    
    def __init__(self, 
                 input_dim: int = 10, 
                 hidden_dims: List[int] = [64, 128], 
                 output_dim: int = 256, 
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = False):
        """
        Initializes the PhysicsAwareMLP.
        
        :param input_dim: Dimension of the physical parameter vector p (default 10, 
                          matching the output of PhysicalParameterAggregator).
        :param hidden_dims: List of dimensions for the hidden layers.
        :param output_dim: Dimension of the output latent physical embedding v_phys (d_phys).
        :param dropout_rate: Dropout probability for regularization.
        :param use_batch_norm: Whether to include Batch Normalization after each hidden layer.
        """
        super(PhysicsAwareMLP, self).__init__()
        
        if input_dim <= 0:
            raise ValueError("input_dim must be strictly positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be strictly positive.")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError("dropout_rate must be in the range [0.0, 1.0).")
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Build the network layers according to the mathematical formulation
        layers = []
        current_dim = input_dim
        
        # Hidden Layers: W_l * h_{l-1} + b_l -> BatchNorm (optional) -> Swish -> Dropout
        for l, h_dim in enumerate(hidden_dims):
            # Linear transformation: W_l * x + b_l
            layers.append(nn.Linear(current_dim, h_dim))
            
            # Optional Batch Normalization for stable training of physical features
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
                
            # Non-linear activation: sigma(x) = x * sigmoid(x)
            layers.append(SwishActivation())
            
            # Regularization
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
                
            current_dim = h_dim
            
        # Output Layer: W_L * h_{L-1} + b_L (No activation to preserve continuous latent space)
        layers.append(nn.Linear(current_dim, output_dim))
        
        # Combine all layers into a Sequential module
        self.phi_phys = nn.Sequential(*layers)
        
        # Initialize weights using Xavier Uniform for better gradient flow
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights (W_l) and biases (b_l) of the linear layers 
        using Xavier Uniform initialization to ensure stable gradient flow 
        during the training of the orthogonal dual-stream architecture.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing the mathematical formulation:
        v_phys = phi_phys(p) = W_L * sigma( ... sigma(W_1 * p + b_1) ... ) + b_L
        
        :param p: Continuous physical parameter vector tensor of shape (batch_size, input_dim).
        :return: Latent physical embedding v_phys of shape (batch_size, output_dim).
        """
        if p.dim() != 2:
            raise ValueError(f"Expected input tensor p to be 2D (batch_size, input_dim), got {p.dim()}D.")
        if p.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {p.shape[1]}.")
            
        # Pass through the sequential MLP
        v_phys = self.phi_phys(p)
        
        return v_phys

    def get_output_dim(self) -> int:
        """
        Returns the dimension of the output latent physical embedding (d_phys).
        Useful for configuring the downstream AdaLN modules in the UNet.
        
        :return: Integer representing d_phys.
        """
        return self.output_dim


if __name__ == "__main__":
    # Example usage and validation of the PhysicsAwareMLP
    
    # 1. Define hyperparameters
    batch_size = 16
    input_dim = 10      # Matches PhysicalParameterAggregator output
    hidden_dims = [64, 128, 256]
    output_dim = 512    # d_phys, matching UNet conditioning dimension
    
    # 2. Instantiate the Physics-aware MLP
    phi_phys = PhysicsAwareMLP(
        input_dim=input_dim, 
        hidden_dims=hidden_dims, 
        output_dim=output_dim, 
        dropout_rate=0.1,
        use_batch_norm=True
    )
    
    print(f"Initialized PhysicsAwareMLP (phi_phys):")
    print(f"  Input Dimension (d): {input_dim}")
    print(f"  Hidden Dimensions: {hidden_dims}")
    print(f"  Output Dimension (d_phys): {phi_phys.get_output_dim()}")
    print(f"  Total Parameters: {sum(p.numel() for p in phi_phys.parameters()):,}")
    
    # 3. Create a orig continuous physical parameter vector p
    # Simulating a batch of physical parameters extracted from histopathology patches
    p_orig = torch.randn(batch_size, input_dim)
    
    # 4. Forward pass to compute v_phys
    v_phys = phi_phys(p_orig)
    
    print(f"\nForward Pass Validation:")
    print(f"  Input p shape: {p_orig.shape}")
    print(f"  Output v_phys shape: {v_phys.shape}")
    
    # 5. Verify output shape and gradient flow
    assert v_phys.shape == (batch_size, output_dim), "Output shape mismatch!"
    
    # Test backward pass (gradient flow)
    loss = v_phys.sum()
    loss.backward()
    
    # Check if gradients exist for the first layer's weights (W_1)
    first_linear_layer = phi_phys.phi_phys[0]
    assert first_linear_layer.weight.grad is not None, "Gradients did not flow back to W_1!"
    
    print("\nVerification passed: Output shape is correct and gradients flow successfully through phi_phys.")