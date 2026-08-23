"""
data_pipeline.py
This script encapsulates the data downloading and preprocessing pipeline for the PhyBio-ODM framework.
It is structured to map to the following directory structure:
├── data/
│   ├── download_tcga.py       (Mapped to TCGAUTDownloader)
│   ├── download_camelyon17.py (Mapped to Camelyon17Downloader)
│   ├── download_paip.py       (Mapped to PAIPDownloader)
│   ├── download_nct.py        (Mapped to NCTCRCHEDownloader)
│   └── preprocessing/         (Mapped to HistopathologyPreprocessor)
"""

import os
import urllib.request
import tarfile
import zipfile
import numpy as np
import cv2
from scipy.linalg import svd
from scipy.optimize import nnls
from typing import List, Tuple, Optional

# ==========================================
# 1. BASE DOWNLOADER CLASS
# ==========================================
class BaseDatasetDownloader:
    """
    Base class for downloading and extracting histopathology datasets.
    Implements the core logic for file retrieval and archive extraction.
    """
    def __init__(self, dataset_name: str, url: str, download_dir: str):
        self.dataset_name = dataset_name
        self.url = url
        self.download_dir = download_dir
        self.raw_dir = os.path.join(download_dir, dataset_name, "raw")
        self.processed_dir = os.path.join(download_dir, dataset_name, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def download_file(self, filename: str) -> str:
        """Downloads the dataset archive if it doesn't already exist."""
        filepath = os.path.join(self.raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"[{self.dataset_name}] Downloading dataset...")
            urllib.request.urlretrieve(self.url, filepath)
            print(f"[{self.dataset_name}] Download complete.")
        else:
            print(f"[{self.dataset_name}] Archive already exists.")
        return filepath

    def extract_archive(self, filepath: str) -> str:
        """Extracts .tar.gz or .zip archives."""
        extract_dir = os.path.join(self.raw_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        if filepath.endswith('.tar.gz') or filepath.endswith('.tgz'):
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=extract_dir)
        elif filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(path=extract_dir)
        else:
            raise ValueError("Unsupported archive format.")
            
        return extract_dir

    def download_and_extract(self) -> str:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement download_and_extract")


# ==========================================
# 2. DATASET-SPECIFIC DOWNLOADERS
# ==========================================
class TCGAUTDownloader(BaseDatasetDownloader):
    """Downloader for TCGA-Universal Tumor (TCGA-UT) dataset."""
    def __init__(self, download_dir: str):
        # Placeholder URL for TCGA-UT
        super().__init__("TCGA_UT", "https://example.com/tcga_ut.zip", download_dir)

    def download_and_extract(self) -> str:
        filepath = self.download_file("tcga_ut.zip")
        return self.extract_archive(filepath)


class Camelyon17Downloader(BaseDatasetDownloader):
    """Downloader for CAMELYON17 dataset."""
    def __init__(self, download_dir: str):
        # Placeholder URL for CAMELYON17
        super().__init__("CAMELYON17", "https://example.com/camelyon17.tar.gz", download_dir)

    def download_and_extract(self) -> str:
        filepath = self.download_file("camelyon17.tar.gz")
        return self.extract_archive(filepath)


class PAIPDownloader(BaseDatasetDownloader):
    """Downloader for PAIP 2019 dataset."""
    def __init__(self, download_dir: str):
        # Placeholder URL for PAIP 2019
        super().__init__("PAIP2019", "https://example.com/paip2019.zip", download_dir)

    def download_and_extract(self) -> str:
        filepath = self.download_file("paip2019.zip")
        return self.extract_archive(filepath)


class NCTCRCHEDownloader(BaseDatasetDownloader):
    """Downloader for NCT-CRC-HE dataset."""
    def __init__(self, download_dir: str):
        # Placeholder URL for NCT-CRC-HE
        super().__init__("NCT_CRC_HE", "https://example.com/nct_crc_he.zip", download_dir)

    def download_and_extract(self) -> str:
        filepath = self.download_file("nct_crc_he.zip")
        return self.extract_archive(filepath)


# ==========================================
# 3. PREPROCESSING & PHYSICAL EXTRACTION
# ==========================================
class HistopathologyPreprocessor:
    """
    Preprocesses downloaded histopathology images into patches and extracts 
    continuous physical parameters based on the proposed methodology's mathematical equations.
    """
    def __init__(self, patch_size: int = 256, stride: int = 256, epsilon: float = 1e-5, I0: float = 1.0):
        self.patch_size = patch_size
        self.stride = stride
        self.epsilon = epsilon
        self.I0 = I0

    def extract_patches(self, image: np.ndarray) -> List[np.ndarray]:
        """Extracts non-overlapping patches from a Whole Slide Image (WSI) or large image."""
        h, w, c = image.shape
        patches = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                # Filter out background patches (e.g., mostly white glass)
                if np.mean(patch) < 240: 
                    patches.append(patch)
        return patches

    def compute_optical_density(self, patch: np.ndarray) -> np.ndarray:
        """
        Computes Optical Density (OD) matrix using the Beer-Lambert law:
        D = -log10((I + epsilon) / I0)
        """
        I = patch.astype(np.float32) / 255.0
        D = -np.log10((I + self.epsilon) / self.I0)
        return D

    def estimate_stain_vectors_svd(self, D: np.ndarray, percentile: float = 99.0) -> np.ndarray:
        """
        Estimates stain basis matrix M using Singular Value Decomposition (SVD) 
        on the top percentile of optical density vectors.
        """
        h, w, c = D.shape
        D_reshaped = D.reshape(-1, c)
        
        # Select pixels with high optical density (top percentile)
        od_sum = np.sum(D_reshaped, axis=1)
        threshold = np.percentile(od_sum, percentile)
        high_od_pixels = D_reshaped[od_sum >= threshold]
        
        if high_od_pixels.shape[0] < 3:
            return np.eye(3, 2) 
            
        mean_vec = np.mean(high_od_pixels, axis=0)
        centered_data = high_od_pixels - mean_vec
        
        U, S, Vt = svd(centered_data, full_matrices=False)
        
        # The first two principal components represent Hematoxylin and Eosin
        M = Vt[:2, :].T 
        
        # Ensure consistent orientation (positive values for H and E)
        if M[1, 0] < 0: M[:, 0] = -M[:, 0]
        if M[1, 1] < 0: M[:, 1] = -M[:, 1]
        
        return M

    def unmix_stains_nnls(self, D: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Recovers stain concentration matrix C via Non-Negative Least Squares (NNLS):
        C = argmin_{C' >= 0} || D - C' M^T ||_F^2
        """
        h, w, c = D.shape
        D_reshaped = D.reshape(-1, c)
        C = np.zeros((h * w, M.shape[1]), dtype=np.float32)
        
        for i in range(h * w):
            C[i, :], _ = nnls(M, D_reshaped[i, :])
            
        return C.reshape(h, w, M.shape[1])

    def aggregate_physical_parameters(self, C: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Aggregates continuous physical parameter vector p:
        p = [mu(C), sigma(C), kappa(C), theta_HE, ||m_H||_2, ||m_E||_2, det(M^T M)]^T
        """
        mu_C = np.mean(C, axis=(0, 1))
        sigma_C = np.std(C, axis=(0, 1))
        
        # Kurtosis (4th standardized moment)
        kappa_C = np.mean(((C - mu_C) / (sigma_C + self.epsilon))**4, axis=(0, 1))
        
        m_H, m_E = M[:, 0], M[:, 1]
        theta_HE = np.arccos(np.clip(np.dot(m_H, m_E) / (np.linalg.norm(m_H) * np.linalg.norm(m_E) + self.epsilon), -1.0, 1.0))
        
        norm_mH = np.linalg.norm(m_H)
        norm_mE = np.linalg.norm(m_E)
        det_MT_M = np.linalg.det(M.T @ M)
        
        p = np.concatenate([mu_C, sigma_C, kappa_C, [theta_HE, norm_mH, norm_mE, det_MT_M]])
        return p

    def process_image(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Full pipeline: Patch extraction -> OD -> SVD -> NNLS -> Parameter Aggregation"""
        patches = self.extract_patches(image)
        processed_patches = []
        physical_params = []
        
        for patch in patches:
            D = self.compute_optical_density(patch)
            M = self.estimate_stain_vectors_svd(D)
            C = self.unmix_stains_nnls(D, M)
            p = self.aggregate_physical_parameters(C, M)
            
            processed_patches.append(patch)
            physical_params.append(p)
            
        return processed_patches, physical_params


# ==========================================
# 4. PIPELINE MANAGER
# ==========================================
class DataPipelineManager:
    """Manages the downloading and preprocessing of all datasets."""
    def __init__(self, base_download_dir: str = "./data"):
        self.base_dir = base_download_dir
        self.downloaders = {
            "TCGA_UT": TCGAUTDownloader(self.base_dir),
            "CAMELYON17": Camelyon17Downloader(self.base_dir),
            "PAIP2019": PAIPDownloader(self.base_dir),
            "NCT_CRC_HE": NCTCRCHEDownloader(self.base_dir)
        }
        self.preprocessor = HistopathologyPreprocessor(patch_size=256, stride=256)

    def run_pipeline(self, dataset_name: str, sample_image_path: Optional[str] = None):
        """Downloads the dataset and processes a sample image if provided."""
        if dataset_name not in self.downloaders:
            raise ValueError(f"Dataset {dataset_name} not supported.")
        
        downloader = self.downloaders[dataset_name]
        print(f"\n--- Initiating Pipeline for {dataset_name} ---")
        
        try:
            extract_dir = downloader.download_and_extract()
            print(f"Data extracted to: {extract_dir}")
        except Exception as e:
            print(f"Download/Extraction failed (expected for placeholder URLs): {e}")
            print("Proceeding to preprocessing demonstration...")
        
        if sample_image_path and os.path.exists(sample_image_path):
            print(f"Processing sample image: {sample_image_path}")
            image = cv2.imread(sample_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            patches, params = self.preprocessor.process_image(image)
            print(f"Extracted {len(patches)} valid tissue patches.")
            if params:
                print(f"Continuous Physical Parameter Vector (p) shape: {np.array(params).shape}")
                print(f"First patch p: {params[0]}")
        else:
            print("No sample image provided for preprocessing.")


if __name__ == "__main__":
    # Initialize the pipeline manager
    pipeline = DataPipelineManager(base_download_dir="./data")
    
    # Create a image to test the mathematical preprocessing pipeline
    image = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite("histology_patch.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Run pipeline for TCGA-UT with the image
    pipeline.run_pipeline("TCGA_UT", sample_image_path="histology_patch.png")
    
    # Clean up image
    if os.path.exists("histology_patch.png"):
        os.remove("histology_patch.png")