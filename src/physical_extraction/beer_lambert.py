"""
beer_lambert.py
Implements the Beer-Lambert law for transforming RGB intensity into Optical Density (OD) space.
This is the foundational step for Continuous Physical Parameter Extraction in PhyBio-ODM,
resolving the discrete token bottleneck by grounding the representation in tissue optics.
"""

import numpy as np
from typing import Union

class BeerLambertTransformer:
    """
    Transforms RGB histopathology images into the Optical Density (OD) space 
    using the Beer-Lambert law of light absorption.
    
    Mathematical Formulation:
    1. Intensity Normalization: I_norm = I_raw / 255.0
    2. Beer-Lambert Transformation: D = -log10((I_norm + epsilon) / I0)
    
    Where:
    - I_raw is the input RGB image in uint8 format [0, 255].
    - I_norm is the normalized intensity in [0.0, 1.0].
    - epsilon is a small constant to prevent log(0) and division by zero.
    - I0 is the incident light intensity (typically 1.0 for normalized images).
    - D is the resulting Optical Density matrix.
    """
    
    def __init__(self, epsilon: float = 1e-5, I0: float = 1.0):
        """
        Initializes the BeerLambertTransformer with physical constants.
        
        :param epsilon: Small constant for numerical stability (prevents log(0)).
        :param I0: Incident light intensity (default is 1.0 for normalized [0,1] space).
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be strictly positive to ensure numerical stability.")
        if I0 <= 0:
            raise ValueError("Incident light intensity (I0) must be strictly positive.")
            
        self.epsilon = epsilon
        self.I0 = I0

    def normalize_intensity(self, I_raw: np.ndarray) -> np.ndarray:
        """
        Normalizes raw RGB image intensities from uint8 [0, 255] to float32 [0.0, 1.0].
        
        Equation: I_norm = I_raw / 255.0
        
        :param I_raw: Input RGB image of shape (H, W, 3) and dtype uint8.
        :return: Normalized intensity matrix of shape (H, W, 3) and dtype float32.
        """
        if I_raw.dtype != np.uint8:
            raise TypeError(f"Expected input image of dtype uint8, got {I_raw.dtype}")
            
        I_norm = I_raw.astype(np.float32) / 255.0
        return I_norm

    def compute_optical_density(self, I_norm: np.ndarray) -> np.ndarray:
        """
        Computes the Optical Density (OD) matrix from normalized intensities.
        
        Equation: D = -log10((I_norm + epsilon) / I0)
        
        :param I_norm: Normalized intensity matrix of shape (H, W, 3) in [0.0, 1.0].
        :return: Optical Density matrix D of shape (H, W, 3).
        """
        if I_norm.dtype != np.float32:
            raise TypeError(f"Expected normalized input of dtype float32, got {I_norm.dtype}")
            
        # Apply the Beer-Lambert transformation
        # D = -log10((I + epsilon) / I0)
        numerator = I_norm + self.epsilon
        denominator = self.I0
        
        ratio = numerator / denominator
        D = -np.log10(ratio)
        
        return D

    def transform(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Full pipeline to transform a raw RGB image into the Optical Density space.
        Reuses the normalization and OD computation methods.
        
        :param image_rgb: Input RGB image of shape (H, W, 3) and dtype uint8.
        :return: Optical Density matrix D of shape (H, W, 3).
        """
        I_norm = self.normalize_intensity(image_rgb)
        D = self.compute_optical_density(I_norm)
        return D

    def inverse_transform(self, D: np.ndarray) -> np.ndarray:
        """
        Reconstructs the normalized RGB intensity from the Optical Density space.
        Useful for verifying the physical consistency of the generated OD matrices.
        
        Equation: I_norm = I0 * 10^(-D) - epsilon
        
        :param D: Optical Density matrix of shape (H, W, 3).
        :return: Reconstructed normalized intensity matrix of shape (H, W, 3) clipped to [0.0, 1.0].
        """
        # I_norm = I0 * 10^(-D) - epsilon
        I_reconstructed = self.I0 * np.power(10.0, -D) - self.epsilon
        
        # Clip to valid intensity range [0.0, 1.0] to handle any numerical drift
        I_reconstructed = np.clip(I_reconstructed, 0.0, 1.0)
        
        return I_reconstructed


if __name__ == "__main__":
    # Example usage and validation
    transformer = BeerLambertTransformer(epsilon=1e-5, I0=1.0)
    
    # Create a orig RGB patch (e.g., a purple Hematoxylin-like pixel and a pink Eosin-like pixel)
    orig_image = np.zeros((2, 2, 3), dtype=np.uint8)
    orig_image[0, 0] = [100, 50, 150]  # Purple (High H, Low E)
    orig_image[0, 1] = [220, 150, 180] # Pink (Low H, High E)
    orig_image[1, 0] = [240, 240, 240] # Light background
    orig_image[1, 1] = [50, 20, 50]    # Very dark tissue
    
    # Transform to Optical Density
    D = transformer.transform(orig_image)
    
    print("Original RGB Image (uint8):")
    print(orig_image)
    
    print("\nOptical Density Matrix (float32):")
    print(np.round(D, 4))
    
    # Verify inverse transform
    I_reconstructed = transformer.inverse_transform(D)
    I_reconstructed_uint8 = (I_reconstructed * 255).astype(np.uint8)
    
    print("\nReconstructed RGB Image (uint8):")
    print(I_reconstructed_uint8)
    
    # Check reconstruction error
    error = np.abs(orig_image.astype(np.float32) - I_reconstructed_uint8.astype(np.float32))
    print(f"\nMax Reconstruction Error: {np.max(error)}")