"""
stain_unmixing.py
Implements stain unmixing using the Macenko method (SVD) and Non-Negative Least Squares (NNLS).
This module resolves the discrete token bottleneck by extracting continuous physical parameters
directly from the tissue optics, enabling robust zero-shot generalization in PhyBio-ODM.

Mathematical Formulations:
1. Linear Unmixing Model: D ≈ C * M^T
2. Stain Basis Estimation (SVD): M derived from the top percentile of OD pixels.
3. Concentration Recovery (NNLS): C = argmin_{C' >= 0} || D - C' * M^T ||_F^2
"""

import numpy as np
from scipy.linalg import svd
from scipy.optimize import nnls
from typing import Tuple

class StainUnmixer:
    """
    Performs optical density stain unmixing to separate Hematoxylin and Eosin concentrations.
    Uses Singular Value Decomposition (SVD) for stain basis estimation and 
    Non-Negative Least Squares (NNLS) for concentration recovery.
    """
    
    def __init__(self, percentile: float = 99.0):
        """
        Initializes the StainUnmixer.
        
        :param percentile: Percentile of optical density used to select pixels for SVD.
                           Typically 99.0 to focus on the darkest tissue pixels (stains).
        """
        if not (0.0 < percentile <= 100.0):
            raise ValueError("Percentile must be strictly between 0 and 100.")
        self.percentile = percentile

    def select_high_od_pixels(self, D: np.ndarray) -> np.ndarray:
        """
        Selects pixels with high optical density (top `percentile`).
        These pixels contain the most reliable information about the stain absorption vectors.
        
        :param D: Optical Density matrix of shape (H, W, 3).
        :return: Array of high OD pixels of shape (K, 3), where K is the number of selected pixels.
        """
        H, W, C = D.shape
        pixels = D.reshape(-1, C)
        
        # Calculate optical density sum for each pixel (total absorption)
        od_sum = np.sum(pixels, axis=1)
        
        # Threshold at the given percentile
        threshold = np.percentile(od_sum, self.percentile)
        high_od_pixels = pixels[od_sum >= threshold]
        
        if high_od_pixels.shape[0] < 10:
            raise ValueError(f"Not enough high-OD pixels ({high_od_pixels.shape[0]}) to perform SVD. "
                             "Check image content or lower the percentile.")
            
        return high_od_pixels

    def estimate_stain_basis_svd(self, high_od_pixels: np.ndarray) -> np.ndarray:
        """
        Estimates the stain basis matrix M using Singular Value Decomposition (SVD).
        
        Process:
        1. Center the high OD data by subtracting the mean.
        2. Apply SVD to find the principal components.
        3. Extract the first two right singular vectors as the stain basis.
        4. Apply robust angular projection (sign correction) to ensure consistent H/E assignment.
        
        :param high_od_pixels: Array of high OD pixels of shape (K, 3).
        :return: Stain basis matrix M of shape (3, 2), where columns are [m_H, m_E].
        """
        # Center the data
        mean_vec = np.mean(high_od_pixels, axis=0)
        centered_data = high_od_pixels - mean_vec
        
        # Perform SVD: centered_data = U * S * Vt
        U, S, Vt = svd(centered_data, full_matrices=False)
        
        # The first two rows of Vt (transposed) represent the principal stain vectors
        M = Vt[:2, :].T  # Shape: (3, 2)
        
        # --- Robust Angular Projection / Sign Correction ---
        # Ensure the angle between the two stain vectors is acute (dot product > 0)
        # This prevents the vectors from pointing in opposite hemispheres
        if np.dot(M[:, 0], M[:, 1]) < 0:
            M[:, 1] = -M[:, 1]
            
        # Ensure consistent assignment of Hematoxylin (H) and Eosin (E)
        # Hematoxylin typically has higher absorption in the Red channel (index 0)
        # If the first vector has less red absorption than the second, swap them
        if M[0, 0] < M[0, 1]:
            M = M[:, ::-1]  # Swap columns
            
        return M.astype(np.float32)

    def unmix_concentrations_nnls(self, D: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Recovers the stain concentration matrix C via Non-Negative Least Squares (NNLS).
        
        Equation: C = argmin_{C' >= 0} || D - C' * M^T ||_F^2
        
        This ensures that stain concentrations are physically meaningful (non-negative).
        
        :param D: Optical Density matrix of shape (H, W, 3).
        :param M: Stain basis matrix of shape (3, 2).
        :return: Concentration matrix C of shape (H, W, 2) for [Hematoxylin, Eosin].
        """
        H, W, C_channels = D.shape
        pixels = D.reshape(-1, C_channels)
        N = pixels.shape[0]
        
        # Initialize concentration matrix
        C = np.zeros((N, M.shape[1]), dtype=np.float32)
        
        # Solve NNLS for each pixel
        # scipy.optimize.nnls solves: argmin_x || Ax - b ||_2 for x >= 0
        # Here A = M (stain basis), b = pixel_OD, x = pixel_concentration
        for i in range(N):
            C[i, :], _ = nnls(M, pixels[i])
            
        return C.reshape(H, W, M.shape[1])

    def unmix_stains(self, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full pipeline for stain unmixing: Select high OD pixels -> SVD -> NNLS.
        Reuses the individual mathematical methods defined in this class.
        
        :param D: Optical Density matrix of shape (H, W, 3).
        :return: Tuple containing Concentration matrix C (H, W, 2) and Stain basis matrix M (3, 2).
        """
        high_od_pixels = self.select_high_od_pixels(D)
        M = self.estimate_stain_basis_svd(high_od_pixels)
        C = self.unmix_concentrations_nnls(D, M)
        
        return C, M


if __name__ == "__main__":
    # Example usage and validation
    unmixer = StainUnmixer(percentile=99.0)
    
    # Create a orig Optical Density matrix (simulating an H&E stained patch)
    # Shape: (10, 10, 3)
    orig_D = np.random.rand(10, 10, 3).astype(np.float32) * 0.5
    
    # Inject some strong Hematoxylin (purple/blue: high Red, low Green, high Blue absorption)
    # Note: In OD space, higher value means more absorption (darker)
    orig_D[2:5, 2:5, 0] += 0.8  # Red channel
    orig_D[2:5, 2:5, 2] += 0.6  # Blue channel
    
    # Inject some strong Eosin (pink: high Red, high Green absorption)
    orig_D[6:9, 6:9, 0] += 0.7  # Red channel
    orig_D[6:9, 6:9, 1] += 0.7  # Green channel
    
    # Perform stain unmixing
    C, M = unmixer.unmix_stains(orig_D)
    
    print(f"Optical Density Matrix D shape: {orig_D.shape}")
    print(f"Estimated Stain Basis M shape: {M.shape}")
    print(f"Stain Basis Matrix M (columns are m_H, m_E):\n{M}")
    print(f"Concentration Matrix C shape: {C.shape}")
    
    # Verify non-negativity of concentrations
    assert np.all(C >= 0), "Concentrations must be non-negative!"
    print("\nVerification passed: All stain concentrations are non-negative.")