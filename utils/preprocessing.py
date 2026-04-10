import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class DRPreprocessor:
    def __init__(self, target_size=(224, 224), is_training=False):
        self.target_size = target_size
        self.is_training = is_training
        
        # Standard ImageNet normalization for backbone architectures
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Default conversion transform
        self.base_transforms = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

    def apply_clahe(self, image_np, clipLimit=2.0):
        """Applies Contrast Limited Adaptive Histogram Equalization for fundus images."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE only to L-channel
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        
        # Merge back
        merged = cv2.merge((cl, a, b))
        final_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return final_image

    def apply_gaussian_blur(self, image_np):
        """Applies Gaussian blur for noise reduction."""
        return cv2.GaussianBlur(image_np, (3, 3), 0)

    def compute_quality(self, image_np):
        """
        Lightweight heuristic quality estimator.

        Returns:
            quality_score: float in [0,1] (higher is better)
            quality_label: 'high' | 'medium' | 'low'
            metrics: raw intermediate measures for debugging/UI
        """
        # Heuristics are computed on the resized (but otherwise unenhanced) RGB image.
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Blur estimate: variance of Laplacian (higher => sharper)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        blur_var = float(lap.var())
        blur_score = blur_var / (blur_var + 200.0)

        # Noise estimate: std of high-frequency residual
        smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        residual = gray.astype(np.float32) - smooth.astype(np.float32)
        noise_std = float(residual.std())
        noise_norm = noise_std / 25.0
        noise_score = 1.0 / (1.0 + noise_norm)

        # Illumination estimate: brightness closeness + contrast amount
        mean_gray = float(gray.mean())
        std_gray = float(gray.std())
        contrast_score = std_gray / (std_gray + 40.0)
        brightness_score = float(np.exp(-((mean_gray - 128.0) ** 2) / (2 * (50.0 ** 2))))
        illum_score = 0.6 * contrast_score + 0.4 * brightness_score

        quality_score = 0.5 * blur_score + 0.2 * noise_score + 0.3 * illum_score
        quality_score = float(np.clip(quality_score, 0.0, 1.0))

        if quality_score >= 0.7:
            quality_label = "high"
        elif quality_score >= 0.4:
            quality_label = "medium"
        else:
            quality_label = "low"

        return {
            "quality_score": quality_score,
            "quality_label": quality_label,
            "metrics": {
                "blur_var_laplacian": blur_var,
                "noise_std_residual": noise_std,
                "mean_gray": mean_gray,
                "std_gray": std_gray,
            },
        }

    def preprocess(self, image_path_or_pil):
        """Full image preprocessing pipeline."""
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil.convert("RGB")
            
        # Resize safely using PIL
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy for OpenCV
        image_np = np.array(image)

        # Estimate input quality so we can adapt enhancement steps.
        quality = self.compute_quality(image_np)
        is_low_quality = quality["quality_label"] == "low"

        # Step 1: Denoise / enhance contrast
        if is_low_quality:
            # For low-quality/blurry/noisy inputs, aggressive Gaussian blur can
            # remove lesion details; prefer edge-preserving denoising.
            image_np = cv2.bilateralFilter(image_np, d=5, sigmaColor=50, sigmaSpace=50)
            image_np = self.apply_clahe(image_np, clipLimit=1.0)
        else:
            image_np = self.apply_clahe(image_np, clipLimit=2.0)
            image_np = self.apply_gaussian_blur(image_np)
        
        # Standardize colors (done indirectly by ImageNet normalization later)
        
        # Convert back to PIL
        enhanced_image = Image.fromarray(image_np)
        
        # Setup as tensor
        tensor = self.base_transforms(enhanced_image).unsqueeze(0) # added batch dim [1, 3, 224, 224]

        return tensor, enhanced_image, quality
