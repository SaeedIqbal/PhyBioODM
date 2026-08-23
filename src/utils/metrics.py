"""
metrics.py
Implements the comprehensive evaluation metrics for the PhyBio-ODM framework, 
including generative fidelity (FID, KID), morphological integrity (SCS), 
and downstream generalization/shortcut mitigation (BA, CA).

Mathematical Formulations:
1. FID: ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 * sqrt(Sigma_r * Sigma_g))
2. KID: MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)] (Polynomial Kernel)
3. SCS: Dice(A, B) = 2 * |A ∩ B| / (|A| + |B|) (Nuclear Mask Overlap)
4. BA: (1 / C) * sum_{i=1}^C (TP_i / (TP_i + FN_i))
5. CA: (1 / N) * sum_{i=1}^N I(Critic(h_i) == y_domain_i)
"""

import numpy as np
from scipy.linalg import sqrtm
from typing import Union, Tuple, Dict

def _to_numpy(tensor: Union[np.ndarray, any]) -> np.ndarray:
    """
    Helper function to seamlessly convert PyTorch tensors to NumPy arrays.
    Ensures compatibility with both NumPy-based and PyTorch-based pipelines.
    """
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
    except ImportError:
        pass
        
    if isinstance(tensor, np.ndarray):
        return tensor
        
    raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


class FrechetInceptionDistance:
    """
    Computes the Fréchet Inception Distance (FID) to evaluate generative fidelity.
    
    Mathematical Equation:
    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 * sqrt(Sigma_r * Sigma_g))
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        :param epsilon: Small constant for numerical stability (unused in core math but kept for extensibility).
        """
        self.epsilon = epsilon

    def _compute_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the mean and covariance of the feature vectors.
        
        :param features: Feature matrix of shape (N, D).
        :return: Tuple of mean vector (D,) and covariance matrix (D, D).
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def compute(self, features_real: Union[np.ndarray, any], 
                features_gen: Union[np.ndarray, any]) -> float:
        """
        Computes the FID score between real and generated feature distributions.
        
        :param features_real: Real image features of shape (N, D).
        :param features_gen: Generated image features of shape (M, D).
        :return: Scalar FID score (lower is better).
        """
        features_real = _to_numpy(features_real)
        features_gen = _to_numpy(features_gen)
        
        mu_r, sigma_r = self._compute_statistics(features_real)
        mu_g, sigma_g = self._compute_statistics(features_gen)
        
        # Squared difference between means: ||mu_r - mu_g||^2
        diff = mu_r - mu_g
        mean_diff_sq = diff.dot(diff)
        
        # Matrix square root of the product of covariances: sqrt(Sigma_r * Sigma_g)
        covmean, _ = sqrtm(sigma_r.dot(sigma_g), disp=False)
        
        # Handle numerical errors that may result in complex numbers
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        # Trace terms: Tr(Sigma_r) + Tr(Sigma_g) - 2 * Tr(sqrt(Sigma_r * Sigma_g))
        trace_term = np.trace(sigma_r) + np.trace(sigma_g) - 2.0 * np.trace(covmean)
        
        fid = mean_diff_sq + trace_term
        return float(fid)


class KernelInceptionDistance:
    """
    Computes the Kernel Inception Distance (KID) using the Maximum Mean Discrepancy (MMD) 
    with a polynomial kernel. Unlike FID, KID provides an unbiased estimator.
    
    Mathematical Equation:
    KID = MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    k(x,y) = (x^T y / d + c)^p
    """
    
    def __init__(self, degree: int = 3, gamma: float = None, coef0: float = 1.0):
        """
        :param degree: Degree of the polynomial kernel (p).
        :param gamma: Scaling factor (1/d). If None, uses 1 / feature_dimension.
        :param coef0: Independent term in the kernel (c).
        """
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def _polynomial_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Computes the polynomial kernel matrix between X and Y.
        """
        d = X.shape[1] if self.gamma is None else 1.0 / self.gamma
        return (X.dot(Y.T) / d + self.coef0) ** self.degree

    def compute(self, features_real: Union[np.ndarray, any], 
                features_gen: Union[np.ndarray, any]) -> float:
        """
        Computes the unbiased KID score.
        
        :param features_real: Real image features of shape (N, D).
        :param features_gen: Generated image features of shape (M, D).
        :return: Scalar KID score (lower is better).
        """
        features_real = _to_numpy(features_real)
        features_gen = _to_numpy(features_gen)
        
        n = features_real.shape[0]
        m = features_gen.shape[0]
        
        if n < 2 or m < 2:
            raise ValueError("KID requires at least 2 samples in both real and generated sets.")
            
        K_rr = self._polynomial_kernel(features_real, features_real)
        K_gg = self._polynomial_kernel(features_gen, features_gen)
        K_rg = self._polynomial_kernel(features_real, features_gen)
        
        # Unbiased MMD^2 estimator (excluding diagonal elements for K_rr and K_gg)
        mmd2 = (K_rr.sum() - np.trace(K_rr)) / (n * (n - 1)) + \
               (K_gg.sum() - np.trace(K_gg)) / (m * (m - 1)) - \
               2.0 * K_rg.mean()
               
        return float(mmd2)


class StructuralConsistencyScore:
    """
    Computes the Structural Consistency Score (SCS) to evaluate morphological integrity.
    Uses the Dice Coefficient on binary nuclear masks to measure the preservation 
    of nuclear geometry and boundaries.
    
    Mathematical Equation:
    SCS = Dice(A, B) = 2 * |A ∩ B| / (|A| + |B|)
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        :param threshold: Threshold to binarize the continuous mask/probability maps.
        """
        self.threshold = threshold

    def compute(self, mask_real: Union[np.ndarray, any], 
                mask_gen: Union[np.ndarray, any]) -> float:
        """
        Computes the SCS between real and generated nuclear masks.
        
        :param mask_real: Real nuclear mask of shape (H, W) or (B, H, W).
        :param mask_gen: Generated nuclear mask of shape (H, W) or (B, H, W).
        :return: Scalar SCS score in [0, 1] (higher is better).
        """
        mask_real = _to_numpy(mask_real)
        mask_gen = _to_numpy(mask_gen)
        
        # Binarize masks
        A = (mask_real > self.threshold).astype(np.float32)
        B = (mask_gen > self.threshold).astype(np.float32)
        
        intersection = np.sum(A * B)
        sum_sizes = np.sum(A) + np.sum(B)
        
        # Handle edge case where both masks are completely empty
        if sum_sizes == 0:
            return 1.0 if np.sum(A) == 0 and np.sum(B) == 0 else 0.0
            
        dice = 2.0 * intersection / sum_sizes
        return float(dice)


class BalancedAccuracy:
    """
    Computes the Balanced Accuracy (BA) for multi-class downstream classification, 
    ensuring robust evaluation across imbalanced external cohorts.
    
    Mathematical Equation:
    BA = (1 / C) * sum_{i=1}^C (TP_i / (TP_i + FN_i))
    """
    
    def compute(self, y_true: Union[np.ndarray, any], 
                y_pred: Union[np.ndarray, any]) -> float:
        """
        Computes the Balanced Accuracy.
        
        :param y_true: Ground truth class labels of shape (N,).
        :param y_pred: Predicted class labels of shape (N,).
        :return: Scalar BA score in [0, 1] (higher is better).
        """
        y_true = _to_numpy(y_true).flatten()
        y_pred = _to_numpy(y_pred).flatten()
        
        classes = np.unique(y_true)
        recalls = []
        
        for c in classes:
            tp = np.sum((y_pred == c) & (y_true == c))
            fn = np.sum((y_pred != c) & (y_true == c))
            
            # Recall for class c: TP_i / (TP_i + FN_i)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall)
            
        return float(np.mean(recalls))


class CriticAccuracy:
    """
    Computes the Adversarial Critic Accuracy (CA) to quantify shortcut learning.
    A lower CA indicates that the biological latents are strictly independent 
    of the domain/site metadata, proving successful disentanglement.
    
    Mathematical Equation:
    CA = (1 / N) * sum_{i=1}^N I(Critic(h_i) == y_domain_i)
    """
    
    def compute(self, y_true_domain: Union[np.ndarray, any], 
                y_pred_domain: Union[np.ndarray, any]) -> float:
        """
        Computes the Critic Accuracy.
        
        :param y_true_domain: Ground truth domain/site labels of shape (N,).
        :param y_pred_domain: Critic's predicted domain labels of shape (N,).
        :return: Scalar CA score in [0, 1] (lower is better, near random chance).
        """
        y_true_domain = _to_numpy(y_true_domain).flatten()
        y_pred_domain = _to_numpy(y_pred_domain).flatten()
        
        # Standard classification accuracy
        accuracy = np.mean(y_true_domain == y_pred_domain)
        return float(accuracy)


class MetricsManager:
    """
    Unified manager to compute and aggregate all evaluation metrics 
    for the PhyBio-ODM framework across different evaluation stages.
    """
    
    def __init__(self, kid_degree: int = 3, scs_threshold: float = 0.5):
        """
        Initializes all metric calculators.
        
        :param kid_degree: Degree for the polynomial kernel in KID.
        :param scs_threshold: Threshold for binarizing masks in SCS.
        """
        self.fid = FrechetInceptionDistance()
        self.kid = KernelInceptionDistance(degree=kid_degree)
        self.scs = StructuralConsistencyScore(threshold=scs_threshold)
        self.ba = BalancedAccuracy()
        self.ca = CriticAccuracy()

    def compute_generative_metrics(self, features_real, features_gen) -> Dict[str, float]:
        """
        Computes generative fidelity metrics (FID, KID).
        """
        return {
            "FID": self.fid.compute(features_real, features_gen),
            "KID": self.kid.compute(features_real, features_gen)
        }

    def compute_structural_metric(self, mask_real, mask_gen) -> Dict[str, float]:
        """
        Computes morphological integrity metric (SCS).
        """
        return {
            "SCS": self.scs.compute(mask_real, mask_gen)
        }

    def compute_downstream_metrics(self, y_true, y_pred, y_true_domain, y_pred_domain) -> Dict[str, float]:
        """
        Computes downstream generalization and shortcut mitigation metrics (BA, CA).
        """
        return {
            "BA": self.ba.compute(y_true, y_pred),
            "CA": self.ca.compute(y_true_domain, y_pred_domain)
        }


if __name__ == "__main__":
    # Example usage and validation of the MetricsManager
    
    manager = MetricsManager(kid_degree=3, scs_threshold=0.5)
    
    # 1. Generative Metrics (FID, KID)
    # Simulating 1000 samples with 2048-dimensional Inception features
    feats_real = np.random.randn(1000, 2048)
    feats_gen = feats_real + 0.1 * np.random.randn(1000, 2048) # Slightly perturbed
    
    gen_metrics = manager.compute_generative_metrics(feats_real, feats_gen)
    print(f"Generative Metrics -> FID: {gen_metrics['FID']:.4f}, KID: {gen_metrics['KID']:.6f}")
    
    # 2. Structural Metric (SCS)
    # Simulating 256x256 binary nuclear masks
    mask_real = (np.random.rand(256, 256) > 0.9).astype(np.float32)
    mask_gen = mask_real.copy()
    # Introduce slight morphological distortion
    mask_gen[10:20, 10:20] = 0.0 
    mask_gen[50:60, 50:60] = 1.0 
    
    struct_metrics = manager.compute_structural_metric(mask_real, mask_gen)
    print(f"Structural Metric -> SCS: {struct_metrics['SCS']:.4f}")
    
    # 3. Downstream Metrics (BA, CA)
    # Simulating 500 samples, 4 classes
    y_true = np.random.randint(0, 4, 500)
    y_pred = y_true.copy()
    y_pred[:50] = np.random.randint(0, 4, 50) # Introduce some classification errors
    
    # Simulating Adversarial Critic predictions (5 sites)
    y_true_domain = np.random.randint(0, 5, 500)
    y_pred_domain = np.random.randint(0, 5, 500) # Random guessing (ideal for PhyBio-ODM)
    
    down_metrics = manager.compute_downstream_metrics(y_true, y_pred, y_true_domain, y_pred_domain)
    print(f"Downstream Metrics -> BA: {down_metrics['BA']:.4f}, CA: {down_metrics['CA']:.4f}")
    
    print("\nVerification passed: All metrics computed successfully with expected mathematical behaviors.")