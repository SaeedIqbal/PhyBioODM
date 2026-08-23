"""
zero_shot_inference.py
Resolves Gap 4: Failure on Unseen Domains.
Extracts continuous physical parameters from unseen hospital reference images 
and maps them to the latent physical embedding for zero-shot synthesis.

Mathematical Formulations:
1. Extraction Operator: p_new = (1/M) * sum_{j=1}^M Psi(I_j)
   where Psi(I) = [mu(C), sigma(C), kappa(C), theta_HE, ||m_H||_2, ||m_E||_2, det(M^T M)]^T
2. Latent Mapping: v_phys_new = phi_phys(p_new)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import svd
from scipy.optimize import nnls
from typing import List, Tuple, Optional
from tqdm import tqdm

class PhysicalExtractionOperator:
    """
    Encapsulates the physical extraction operator Psi(I) to ensure self-contained execution.
    Maps RGB images to continuous physical parameter vectors p using the Beer-Lambert law, 
    SVD stain unmixing, and statistical aggregation.
    """
    def __init__(self, epsilon: float = 1e-5, I0: float = 1.0, percentile: float = 99.0):
        self.epsilon = epsilon
        self.I0 = I0
        self.percentile = percentile

    def compute_optical_density(self, patch: np.ndarray) -> np.ndarray:
        """D = -log10((I + epsilon) / I0)"""
        I = patch.astype(np.float32) / 255.0
        D = -np.log10((I + self.epsilon) / self.I0)
        return D

    def estimate_stain_basis_svd(self, D: np.ndarray) -> np.ndarray:
        """Estimates stain basis matrix M using SVD on the top percentile of OD pixels."""
        H, W, C = D.shape
        pixels = D.reshape(-1, C)
        od_sum = np.sum(pixels, axis=1)
        threshold = np.percentile(od_sum, self.percentile)
        high_od_pixels = pixels[od_sum >= threshold]
        
        if high_od_pixels.shape[0] < 10:
            return np.eye(3, 2, dtype=np.float32)
            
        mean_vec = np.mean(high_od_pixels, axis=0)
        centered_data = high_od_pixels - mean_vec
        U, S, Vt = svd(centered_data, full_matrices=False)
        M = Vt[:2, :].T 
        
        if np.dot(M[:, 0], M[:, 1]) < 0: M[:, 1] = -M[:, 1]
        if M[0, 0] < M[0, 1]: M = M[:, ::-1]
        return M.astype(np.float32)

    def unmix_stains_nnls(self, D: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Recovers stain concentration matrix C via Non-Negative Least Squares (NNLS)."""
        H, W, C = D.shape
        pixels = D.reshape(-1, C)
        C_mat = np.zeros((pixels.shape[0], M.shape[1]), dtype=np.float32)
        for i in range(pixels.shape[0]):
            C_mat[i, :], _ = nnls(M, pixels[i])
        return C_mat.reshape(H, W, M.shape[1])

    def aggregate_parameters(self, C: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Aggregates continuous physical parameter vector p."""
        mu = np.mean(C, axis=(0, 1))
        sigma = np.std(C, axis=(0, 1))
        kappa = np.mean(((C - mu) / (sigma + self.epsilon))**4, axis=(0, 1))
        
        m_H, m_E = M[:, 0], M[:, 1]
        theta_HE = np.arccos(np.clip(np.dot(m_H, m_E) / (np.linalg.norm(m_H) * np.linalg.norm(m_E) + self.epsilon), -1.0, 1.0))
        det_MT_M = np.linalg.det(M.T @ M)
        
        p = np.concatenate([mu, sigma, kappa, [theta_HE, np.linalg.norm(m_H), np.linalg.norm(m_E), det_MT_M]])
        return p.astype(np.float32)

    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        """Full extraction pipeline: I -> D -> M, C -> p"""
        D = self.compute_optical_density(image_rgb)
        M = self.estimate_stain_basis_svd(D)
        C = self.unmix_stains_nnls(D, M)
        p = self.aggregate_parameters(C, M)
        return p


class ZeroShotInferencePipeline:
    """
    Manages the zero-shot inference pipeline for unseen medical domains.
    Extracts continuous physical parameters from a sparse reference set of an unseen hospital 
    and guides the reverse SDE solver to generate synthetic patches.
    """
    def __init__(self, 
                 model: nn.Module, 
                 phi_phys: nn.Module, 
                 sde_solver, 
                 extraction_operator: PhysicalExtractionOperator,
                 dataset_name: str = 'CAMELYON17',
                 device: str = 'cuda'):
        """
        :param model: The trained PhyBio-ODM UNet backbone.
        :param phi_phys: The trained Physics-aware MLP.
        :param sde_solver: The ReverseSDESolver instance.
        :param extraction_operator: The PhysicalExtractionOperator instance.
        :param dataset_name: Target unseen dataset ('CAMELYON17', 'PAIP_2019', 'NCT-CRC-HE').
        :param device: Computation device.
        """
        self.model = model.to(device).eval()
        self.phi_phys = phi_phys.to(device).eval()
        self.sde_solver = sde_solver
        self.extraction_operator = extraction_operator
        self.dataset_name = dataset_name
        self.device = device

    def load_reference_images(self, reference_dir: str, max_samples: int = 50) -> List[np.ndarray]:
        """
        Loads a sparse reference set of images from the unseen hospital.
        Simulates loading from the original datasets (e.g., CAMELYON17 center 4).
        """
        images = []
        if not os.path.exists(reference_dir):
            print(f"Warning: Reference directory {reference_dir} not found. Generating mock data for {self.dataset_name}.")
            # Generate mock H&E-like patches for demonstration
            for _ in range(max_samples):
                mock_img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
                images.append(mock_img)
            return images

        files = sorted([f for f in os.listdir(reference_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])[:max_samples]
        for f in files:
            img_path = os.path.join(reference_dir, f)
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
        return images

    def extract_physical_parameters(self, reference_images: List[np.ndarray]) -> np.ndarray:
        """
        Computes the robust empirical mean of the physics-grounded features.
        Equation: p_new = (1/M) * sum_{j=1}^M Psi(I_j)
        """
        p_list = []
        for img in tqdm(reference_images, desc=f"Extracting physical params from {self.dataset_name}"):
            p_j = self.extraction_operator.extract(img)
            p_list.append(p_j)
            
        p_new = np.mean(np.stack(p_list, axis=0), axis=0)
        return p_new

    @torch.no_grad()
    def generate_zero_shot_samples(self, 
                                   p_new: np.ndarray, 
                                   y: torch.Tensor, 
                                   num_samples: int = 16, 
                                   img_size: int = 256) -> torch.Tensor:
        """
        Generates synthetic patches for the unseen domain using the reverse SDE solver.
        
        :param p_new: Continuous physical parameter vector of the unseen domain (d,).
        :param y: Discrete class label tensor (num_samples,).
        :param num_samples: Number of patches to generate.
        :param img_size: Spatial resolution of the generated patches.
        :return: Generated clean images x_0 of shape (num_samples, 3, img_size, img_size).
        """
        # 1. Map continuous physical parameters to latent space
        # Equation: v_phys_new = phi_phys(p_new)
        p_new_tensor = torch.tensor(p_new, dtype=torch.float32).unsqueeze(0).repeat(num_samples, 1).to(self.device)
        v_phys_new = self.phi_phys(p_new_tensor)
        
        # 2. Get biological embedding
        # Equation: v_bio = E_class(y)
        v_bio = self.model.get_biological_embedding(y.to(self.device))
        
        # 3. Initialize noise x_T
        x_T = torch.randn(num_samples, 3, img_size, img_size, device=self.device)
        
        # 4. Solve reverse SDE
        # Equation: dx_t = [f - g^2 \nabla log p_theta] dt + g dw_t
        x_0 = self.sde_solver.solve(x_T, v_bio, v_phys_new)
        
        return x_0


if __name__ == "__main__":
    # Mock components for demonstration
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 256)
        def get_biological_embedding(self, y): return self.emb(y)
        def predict_noise_with_latents(self, x_t, t, v_bio, v_phys): return torch.randn_like(x_t) * 0.1

    class MockPhiPhys(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(10, 256)
        def forward(self, p): return self.net(p)

    from sde_solver import ReverseSDESolver

    # Initialize pipeline
    model = MockModel()
    phi_phys = MockPhiPhys()
    solver = ReverseSDESolver(model, num_steps=100, device='cpu')
    extractor = PhysicalExtractionOperator()
    
    pipeline = ZeroShotInferencePipeline(model, phi_phys, solver, extractor, dataset_name='CAMELYON17', device='cpu')
    
    # Run zero-shot inference
    ref_images = pipeline.load_reference_images("./mock_camelyon17_center4", max_samples=10)
    p_new = pipeline.extract_physical_parameters(ref_images)
    
    y_orig = torch.randint(0, 4, (4,))
    generated_samples = pipeline.generate_zero_shot_samples(p_new, y_orig, num_samples=4, img_size=64)
    
    print(f"Generated zero-shot samples shape: {generated_samples.shape}")
    print("Zero-shot inference pipeline executed successfully.")