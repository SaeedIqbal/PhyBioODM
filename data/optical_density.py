"""
optical_density.py
Precomputes Optical Density (OD) matrices and performs stain unmixing.
Implements the Beer-Lambert law and SVD-based stain vector estimation.
"""

import numpy as np
from scipy.linalg import svd
from scipy.optimize import nnls
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OpticalDensityPreprocessor:
    """
    Computes Optical Density (OD) matrices and unmixes stains into Hematoxylin and Eosin concentrations.
    
    Mathematical Formulations:
    1. Beer-Lambert Law: D = -log10((I + epsilon) / I0)
    2. Linear Unmixing: D ≈ C * M^T
    3. Stain Estimation (SVD): M derived from top percentile of OD pixels.
    4. Concentration Recovery (NNLS): C = argmin_{C' >= 0} || D - C' * M^T ||_F^2
    """
    
    def __init__(self, epsilon: float = 1e-5, I0: float = 1.0, percentile: float = 99.0):
        """
        Initializes the OpticalDensityPreprocessor.
        
        :param epsilon: Small constant to prevent log(0) and division by zero.
        :param I0: Incident light intensity (usually 1.0 for normalized [0,1] images).
        :param percentile: Percentile of optical density used to select pixels for SVD.
        """
        self.epsilon = epsilon
        self.I0 = I0
        self.percentile = percentile

    def compute_optical_density(self, patch: np.ndarray) -> np.ndarray:
        """
        Transforms an RGB patch into the Optical Density (OD) space.
        
        Equation: D = -log10((I + epsilon) / I0)
        where I is the normalized RGB intensity in [0, 1].
        
        :param patch: RGB image patch (H, W, 3) in uint8 format.
        :return: Optical Density matrix D (H, W, 3) in float32.
        """
        # Normalize to [0, 1]
        I = patch.astype(np.float32) / 255.0
        
        # Apply Beer-Lambert transformation
        D = -np.log10((I + self.epsilon) / self.I0)
        
        return D

    def estimate_stain_basis_svd(self, D: np.ndarray) -> np.ndarray:
        """
        Estimates the stain basis matrix M using Singular Value Decomposition (SVD).
        
        Process:
        1. Select pixels with high optical density (top `percentile`).
        2. Center the data by subtracting the mean.
        3. Apply SVD and extract the first two right singular vectors.
        
        :param D: Optical Density matrix (H, W, 3).
        :return: Stain basis matrix M of shape (3, 2), where columns are [m_H, m_E].
        """
        h, w, c = D.shape
        pixels = D.reshape(-1, c)
        
        # Calculate optical density sum for each pixel to find the darkest ones
        od_sum = np.sum(pixels, axis=1)
        threshold = np.percentile(od_sum, self.percentile)
        
        # Select top percentile pixels
        high_od_pixels = pixels[od_sum >= threshold]
        
        if high_od_pixels.shape[0] < 10:
            logging.warning("Not enough high-OD pixels for SVD. Returning identity-like matrix.")
            return np.eye(3, 2, dtype=np.float32)
            
        # Center the data
        mean_vec = np.mean(high_od_pixels, axis=0)
        centered_data = high_od_pixels - mean_vec
        
        # Perform SVD
        # centered_data = U * S * Vt
        U, S, Vt = svd(centered_data, full_matrices=False)
        
        # The first two rows of Vt (transposed) represent the principal stain vectors
        M = Vt[:2, :].T 
        
        # Heuristic sign correction to ensure consistency (Hematoxylin usually has higher red absorption)
        # This prevents the model from learning flipped stain vectors across different patches
        if M[0, 0] < 0: 
            M[:, 0] = -M[:, 0]
        if M[0, 1] < 0: 
            M[:, 1] = -M[:, 1]
            
        return M.astype(np.float32)

    def unmix_stains_nnls(self, D: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Recovers the stain concentration matrix C via Non-Negative Least Squares (NNLS).
        
        Equation: C = argmin_{C' >= 0} || D - C' * M^T ||_F^2
        
        :param D: Optical Density matrix (H, W, 3).
        :param M: Stain basis matrix (3, 2).
        :return: Concentration matrix C (H, W, 2) for [Hematoxylin, Eosin].
        """
        h, w, c = D.shape
        pixels = D.reshape(-1, c)
        
        # Initialize concentration matrix
        C = np.zeros((pixels.shape[0], M.shape[1]), dtype=np.float32)
        
        # Solve NNLS for each pixel
        for i in range(pixels.shape[0]):
            # scipy.optimize.nnls solves: argmin_x || Ax - b ||_2 for x >= 0
            # Here A = M, b = pixel_OD, x = pixel_concentration
            C[i, :], _ = nnls(M, pixels[i])
            
        return C.reshape(h, w, M.shape[1])

    def process_patch(self, patch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Full pipeline for a single patch: RGB -> OD -> SVD -> NNLS.
        Reuses the individual mathematical methods defined above.
        
        :param patch: RGB image patch (H, W, 3) in uint8.
        :return: Dictionary containing 'D' (OD), 'M' (Stain Basis), and 'C' (Concentrations).
        """
        D = self.compute_optical_density(patch)
        M = self.estimate_stain_basis_svd(D)
        C = self.unmix_stains_nnls(D, M)
        
        return {
            "optical_density": D,
            "stain_basis": M,
            "concentrations": C
        }

    def precompute_batch(self, patches: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """
        Processes a batch of patches and returns their precomputed OD and unmixing results.
        
        :param patches: List of RGB image patches.
        :return: List of dictionaries containing precomputed physical parameters.
        """
        results = []
        for idx, patch in enumerate(patches):
            try:
                result = self.process_patch(patch)
                results.append(result)
            except Exception as e:
                logging.error(f"Failed to process patch {idx}: {str(e)}")
                
        logging.info(f"Successfully precomputed OD and stain unmixing for {len(results)} patches.")
        return results


if __name__ == "__main__":
    # Example usage
    preprocessor = OpticalDensityPreprocessor(epsilon=1e-5, I0=1.0, percentile=99.0)
    
    # Create a orig H&E-like patch for testing
    # Hematoxylin (purple/blue) and Eosin (pink)
    orig_patch = np.ones((256, 256, 3), dtype=np.uint8) * 240 
    orig_patch[50:150, 50:150] = [100, 50, 150]  # Purple (Hematoxylin-like)
    orig_patch[150:200, 150:200] = [220, 150, 180] # Pink (Eosin-like)
    
    # Process the patch
    results = preprocessor.process_patch(orig_patch)
    
    print(f"Optical Density shape: {results['optical_density'].shape}")
    print(f"Stain Basis Matrix M (3x2):\n{results['stain_basis']}")
    print(f"Concentrations shape: {results['concentrations'].shape}")
    print(f"Max Hematoxylin concentration: {np.max(results['concentrations'][:, :, 0]):.4f}")
    print(f"Max Eosin concentration: {np.max(results['concentrations'][:, :, 1]):.4f}")