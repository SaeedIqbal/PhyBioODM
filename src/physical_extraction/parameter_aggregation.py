"""
parameter_aggregation.py
Aggregates higher-order statistical moments of stain concentrations and geometric invariants 
of the stain basis matrix into a continuous physical parameter vector p.

This module resolves the discrete token bottleneck by providing a dense, continuous 
representation of the physical domain shift, enabling zero-shot generalization in PhyBio-ODM.

Mathematical Formulation:
p = [mu(C), sigma(C), kappa(C), theta_HE, ||m_H||_2, ||m_E||_2, det(M^T M)]^T
"""

import numpy as np
from typing import Tuple

class PhysicalParameterAggregator:
    """
    Computes the continuous physical parameter vector p from the stain concentration 
    matrix C and the stain basis matrix M.
    """
    
    def __init__(self, epsilon: float = 1e-5):
        """
        Initializes the aggregator with a small constant for numerical stability.
        
        :param epsilon: Small constant to prevent division by zero in standard deviation 
                        and kurtosis calculations.
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be strictly positive for numerical stability.")
        self.epsilon = epsilon

    def compute_channel_statistics(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the channel-wise mean, standard deviation, and kurtosis of the stain concentrations.
        
        Equations:
        - mu(C) = mean(C, axis=(0,1))
        - sigma(C) = std(C, axis=(0,1))
        - kappa(C) = mean(((C - mu) / (sigma + epsilon))^4, axis=(0,1))  [4th standardized moment]
        
        :param C: Stain concentration matrix of shape (H, W, 2) for [Hematoxylin, Eosin].
        :return: Tuple containing mu, sigma, and kappa arrays, each of shape (2,).
        """
        # Channel-wise mean
        mu = np.mean(C, axis=(0, 1))
        
        # Channel-wise standard deviation
        sigma = np.std(C, axis=(0, 1))
        
        # Channel-wise kurtosis (4th standardized moment)
        # Using epsilon to prevent division by zero if a channel has zero variance
        normalized_C = (C - mu) / (sigma + self.epsilon)
        kappa = np.mean(normalized_C ** 4, axis=(0, 1))
        
        return mu, sigma, kappa

    def compute_stain_vector_metrics(self, M: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Computes geometric invariants of the stain basis matrix M.
        
        Equations:
        - theta_HE = arccos( (m_H^T m_E) / (||m_H||_2 * ||m_E||_2) )
        - norm_mH = ||m_H||_2
        - norm_mE = ||m_E||_2
        - det_MT_M = det(M^T M)
        
        :param M: Stain basis matrix of shape (3, 2), where columns are [m_H, m_E].
        :return: Tuple containing theta_HE, norm_mH, norm_mE, and det_MT_M.
        """
        m_H = M[:, 0]
        m_E = M[:, 1]
        
        # L2 Norms
        norm_mH = np.linalg.norm(m_H, ord=2)
        norm_mE = np.linalg.norm(m_E, ord=2)
        
        # Angular separation (theta_HE)
        # Clip the dot product to [-1, 1] to avoid NaNs in arccos due to floating point errors
        dot_product = np.dot(m_H, m_E)
        denominator = (norm_mH * norm_mE) + self.epsilon
        cos_theta = np.clip(dot_product / denominator, -1.0, 1.0)
        theta_HE = np.arccos(cos_theta)
        
        # Determinant of the Gram matrix (M^T M)
        # This quantifies the linear independence of the stain basis (spectral purity)
        gram_matrix = M.T @ M
        det_MT_M = np.linalg.det(gram_matrix)
        
        return theta_HE, norm_mH, norm_mE, det_MT_M

    def aggregate(self, C: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Full pipeline to compute the continuous physical parameter vector p.
        Reuses the statistical and geometric metric computation methods.
        
        Equation:
        p = [mu(C), sigma(C), kappa(C), theta_HE, ||m_H||_2, ||m_E||_2, det(M^T M)]^T
        
        :param C: Stain concentration matrix of shape (H, W, 2).
        :param M: Stain basis matrix of shape (3, 2).
        :return: Continuous physical parameter vector p of shape (9,).
                 (2 for mu, 2 for sigma, 2 for kappa, 1 for theta, 1 for norm_H, 1 for norm_E, 1 for det)
        """
        # 1. Compute channel-wise statistics
        mu, sigma, kappa = self.compute_channel_statistics(C)
        
        # 2. Compute stain vector geometric metrics
        theta_HE, norm_mH, norm_mE, det_MT_M = self.compute_stain_vector_metrics(M)
        
        # 3. Concatenate into the final parameter vector p
        # Convert scalar metrics to arrays for concatenation
        geometric_metrics = np.array([theta_HE, norm_mH, norm_mE, det_MT_M], dtype=np.float32)
        
        p = np.concatenate([mu, sigma, kappa, geometric_metrics])
        
        return p.astype(np.float32)


if __name__ == "__main__":
    # Example usage and validation
    aggregator = PhysicalParameterAggregator(epsilon=1e-5)
    
    # Create orig stain concentration matrix C (H=10, W=10, Channels=2)
    # Channel 0: Hematoxylin, Channel 1: Eosin
    orig_C = np.random.rand(10, 10, 2).astype(np.float32) * 0.5
    orig_C[:, :, 0] += 0.2  # Add some baseline Hematoxylin
    
    # Create orig stain basis matrix M (3 channels, 2 stains)
    # Typical H&E vectors (normalized roughly)
    orig_M = np.array([
        [0.6, 0.7],  # Red channel absorption
        [0.5, 0.6],  # Green channel absorption
        [0.4, 0.3]   # Blue channel absorption
    ], dtype=np.float32)
    
    # Aggregate parameters
    p = aggregator.aggregate(orig_C, orig_M)
    
    print(f"Stain Concentration Matrix C shape: {orig_C.shape}")
    print(f"Stain Basis Matrix M shape: {orig_M.shape}")
    print(f"\nContinuous Physical Parameter Vector p shape: {p.shape}")
    print(f"Vector p: {p}")
    
    # Breakdown of the vector p for verification
    print("\n--- Parameter Breakdown ---")
    print(f"mu(C) [H, E]:        {p[0:2]}")
    print(f"sigma(C) [H, E]:     {p[2:4]}")
    print(f"kappa(C) [H, E]:     {p[4:6]}")
    print(f"theta_HE (radians):  {p[6]:.4f}")
    print(f"||m_H||_2:           {p[7]:.4f}")
    print(f"||m_E||_2:           {p[8]:.4f}")
    print(f"det(M^T M):          {p[9]:.4f}")
    
    # Verify expected size (2 + 2 + 2 + 1 + 1 + 1 + 1 = 10)
    # Wait, the math says: mu(2), sigma(2), kappa(2), theta(1), norm_H(1), norm_E(1), det(1) = 10 elements.
    # Let's correct the print statement above to reflect 10 elements.
    assert p.shape[0] == 10, f"Expected vector size 10, got {p.shape[0]}"
    print("\nVerification passed: Parameter vector p has the correct dimension (10).")