"""
discrete_embedding.py
Implements the discrete token embedding mechanism used in the MeDi baseline.
This module represents the "Additive Morphological Entanglement" approach, 
where conditioning is performed via linear additive fusion of learnable embeddings.

Mathematical Formulation:
z_cond = E_class(y) + sum_{j=1}^{J} E_meta^{(j)}(m_j)

Where:
- y in {1, ..., K} is the discrete class label (e.g., tumor subtype).
- m_j is the j-th discrete metadata token (e.g., Tissue Source Site, scanner ID).
- E_class and E_meta^{(j)} are learnable embedding matrices mapping discrete tokens to R^d.
- z_cond in R^d is the final conditioning vector, which is then added to the timestep embedding.

This baseline approach forces orthogonal biological and physical factors into a shared 
Euclidean subspace, leading to the morphological entanglement that PhyBio-ODM aims to resolve.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict

class DiscreteEmbedding(nn.Module):
    """
    PyTorch module implementing the discrete additive conditioning mechanism.
    """
    
    def __init__(self, num_classes: int, metadata_info: List[Tuple[str, int]], embed_dim: int):
        """
        Initializes the DiscreteEmbedding module.
        
        :param num_classes: Number of discrete class labels (K).
        :param metadata_info: List of tuples containing metadata name and its number of categories.
                              e.g., [('tissue_source_site', 32), ('scanner_id', 5)]
        :param embed_dim: Dimension of the latent embedding space (d).
        """
        super(DiscreteEmbedding, self).__init__()
        
        if num_classes <= 0:
            raise ValueError("num_classes must be strictly positive.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be strictly positive.")
        if not metadata_info:
            raise ValueError("metadata_info must contain at least one metadata token.")
            
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # E_class: Learnable embedding matrix for the biological class label y
        self.E_class = nn.Embedding(num_embeddings=num_classes, embedding_dim=embed_dim)
        
        # E_meta^{(j)}: Learnable embedding matrices for each discrete metadata token m_j
        self.E_meta = nn.ModuleDict()
        for meta_name, num_categories in metadata_info:
            if num_categories <= 0:
                raise ValueError(f"Number of categories for {meta_name} must be strictly positive.")
            self.E_meta[meta_name] = nn.Embedding(num_embeddings=num_categories, embedding_dim=embed_dim)
            
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the embedding matrices using a normal distribution to ensure 
        stable gradient flow during the training of the additive conditioning baseline.
        """
        nn.init.normal_(self.E_class.weight, mean=0.0, std=0.02)
        for meta_name in self.E_meta:
            nn.init.normal_(self.E_meta[meta_name].weight, mean=0.0, std=0.02)

    def get_class_embedding(self, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the biological class embedding E_class(y).
        
        :param y: Tensor of discrete class labels of shape (batch_size,).
        :return: Tensor of shape (batch_size, embed_dim).
        """
        return self.E_class(y)

    def get_metadata_embeddings(self, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the sum of metadata embeddings sum_{j=1}^{J} E_meta^{(j)}(m_j).
        
        :param metadata: Dictionary mapping metadata names to tensors of discrete tokens.
                         e.g., {'tissue_source_site': tensor([0, 1, ...]), 'scanner_id': tensor([2, 0, ...])}
        :return: Tensor of shape (batch_size, embed_dim) representing the aggregated metadata embedding.
        """
        batch_size = None
        meta_sum = None
        
        for meta_name, m_j in metadata.items():
            if meta_name not in self.E_meta:
                raise KeyError(f"Metadata token '{meta_name}' not found in initialized embeddings.")
                
            # E_meta^{(j)}(m_j)
            embedding_j = self.E_meta[meta_name](m_j)
            
            if batch_size is None:
                batch_size = embedding_j.shape[0]
                meta_sum = torch.zeros_like(embedding_j)
            elif embedding_j.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch for metadata token '{meta_name}'.")
                
            # Accumulate the sum: sum_{j=1}^{J} E_meta^{(j)}(m_j)
            meta_sum = meta_sum + embedding_j
            
        return meta_sum

    def forward(self, y: torch.Tensor, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass implementing the additive conditioning formulation:
        z_cond = E_class(y) + sum_{j=1}^{J} E_meta^{(j)}(m_j)
        
        :param y: Tensor of discrete class labels of shape (batch_size,).
        :param metadata: Dictionary mapping metadata names to tensors of discrete tokens.
        :return: Final conditioning vector z_cond of shape (batch_size, embed_dim).
        """
        if y.dim() != 1:
            raise ValueError(f"Expected class labels y to be 1D (batch_size,), got {y.dim()}D.")
            
        # 1. Compute E_class(y)
        z_class = self.get_class_embedding(y)
        
        # 2. Compute sum_{j=1}^{J} E_meta^{(j)}(m_j)
        z_meta = self.get_metadata_embeddings(metadata)
        
        # 3. Linear additive fusion: z_cond = z_class + z_meta
        z_cond = z_class + z_meta
        
        return z_cond


if __name__ == "__main__":
    # Example usage and validation of the DiscreteEmbedding (MeDi Baseline)
    
    # 1. Define hyperparameters
    batch_size = 8
    num_classes = 32          # e.g., 32 tumor subtypes
    embed_dim = 256           # d = 256
    
    # Metadata configuration: (name, number_of_categories)
    metadata_info = [
        ('tissue_source_site', 40),  # 40 different hospitals/sites
        ('scanner_id', 5)            # 5 different scanner models
    ]
    
    # 2. Instantiate the Discrete Embedding module
    discrete_cond = DiscreteEmbedding(
        num_classes=num_classes, 
        metadata_info=metadata_info, 
        embed_dim=embed_dim
    )
    
    print(f"Initialized DiscreteEmbedding (MeDi Baseline):")
    print(f"  Number of Classes (K): {num_classes}")
    print(f"  Metadata Tokens: {[name for name, _ in metadata_info]}")
    print(f"  Embedding Dimension (d): {embed_dim}")
    print(f"  Total Parameters: {sum(p.numel() for p in discrete_cond.parameters()):,}")
    
    # 3. Create orig inputs
    # Discrete class labels y
    y_orig = torch.randint(0, num_classes, (batch_size,))
    
    # Discrete metadata tokens m_j
    metadata_orig = {
        'tissue_source_site': torch.randint(0, 40, (batch_size,)),
        'scanner_id': torch.randint(0, 5, (batch_size,))
    }
    
    # 4. Forward pass to compute z_cond
    z_cond = discrete_cond(y_orig, metadata_orig)
    
    print(f"\nForward Pass Validation:")
    print(f"  Input y shape: {y_orig.shape}")
    print(f"  Output z_cond shape: {z_cond.shape}")
    
    # 5. Verify output shape and gradient flow
    assert z_cond.shape == (batch_size, embed_dim), "Output shape mismatch!"
    
    # Test backward pass (gradient flow)
    loss = z_cond.sum()
    loss.backward()
    
    # Check if gradients exist for the class embedding matrix (E_class)
    assert discrete_cond.E_class.weight.grad is not None, "Gradients did not flow back to E_class!"
    
    print("\nVerification passed: Output shape is correct and gradients flow successfully through the additive conditioning.")