"""
patch_extraction.py
Extracts 256x256 or 512x512 patches from histopathology images.
Implements tissue detection to filter out background and artifacts.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Generator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PatchExtractor:
    """
    Extracts non-overlapping or overlapping patches from large histopathology images.
    Filters patches based on tissue content to avoid training on blank glass.
    """
    
    def __init__(self, patch_size: int = 256, stride: int = 256, tissue_threshold: float = 0.20):
        """
        Initializes the PatchExtractor.
        
        :param patch_size: Size of the extracted patches (e.g., 256 or 512).
        :param stride: Step size for the sliding window.
        :param tissue_threshold: Minimum ratio of tissue pixels required to keep a patch.
        """
        self.patch_size = patch_size
        self.stride = stride
        self.tissue_threshold = tissue_threshold

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Loads an image from disk and converts it to RGB format.
        
        :param image_path: Path to the image file.
        :return: Image as a numpy array in RGB format.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Load image using OpenCV (reads in BGR)
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to decode image at {image_path}")
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def _is_tissue_patch(self, patch: np.ndarray) -> bool:
        """
        Determines if a patch contains sufficient tissue using Otsu's thresholding.
        
        Mathematical/Heuristic basis: 
        Tissue pixels in H&E are significantly darker than the white background.
        We convert to grayscale and apply Otsu's automatic thresholding.
        
        :param patch: Image patch in RGB format (H, W, 3).
        :return: True if tissue ratio > tissue_threshold, else False.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # Apply Otsu's thresholding (inverted to make tissue white, background black)
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate the ratio of tissue pixels
        tissue_pixels = np.sum(binary_mask > 0)
        total_pixels = binary_mask.size
        tissue_ratio = tissue_pixels / total_pixels
        
        return tissue_ratio > self.tissue_threshold

    def extract_patches(self, image: np.ndarray) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields valid tissue patches from the input image.
        
        :param image: Full RGB image (H, W, 3).
        :yield: Numpy array of shape (patch_size, patch_size, 3).
        """
        h, w, _ = image.shape
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y + self.patch_size, x:x + self.patch_size]
                
                # Filter out background patches
                if self._is_tissue_patch(patch):
                    yield patch

    def process_and_save(self, image_path: str, output_dir: str, prefix: str = "patch") -> int:
        """
        Orchestrates the loading, extraction, and saving of patches.
        
        :param image_path: Path to the source WSI or large image.
        :param output_dir: Directory to save the extracted patches.
        :param prefix: Filename prefix for saved patches.
        :return: Number of patches successfully saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        image = self.load_image(image_path)
        
        saved_count = 0
        for idx, patch in enumerate(self.extract_patches(image)):
            # Save as RGB PNG
            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(output_dir, f"{prefix}_{idx:05d}.png")
            cv2.imwrite(save_path, patch_bgr)
            saved_count += 1
            
        logging.info(f"Extracted and saved {saved_count} tissue patches from {image_path} to {output_dir}")
        return saved_count


if __name__ == "__main__":
    # Example usage
    extractor = PatchExtractor(patch_size=256, stride=256, tissue_threshold=0.15)
    
    # Create a orig image for testing
    orig_img = np.ones((1024, 1024, 3), dtype=np.uint8) * 240 # White background
    orig_img[200:800, 200:800] = np.random.randint(50, 200, (600, 600, 3), dtype=np.uint8) # Tissue region
    cv2.imwrite("orig_wsi.png", cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
    
    # Run extraction
    extractor.process_and_save("orig_wsi.png", "./orig_patches", prefix="test")
    
    # Cleanup
    os.remove("orig_wsi.png")