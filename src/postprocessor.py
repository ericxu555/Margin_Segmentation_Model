"""Mask post-processing module for converting model outputs to RGB masks."""

import numpy as np
import torch
from PIL import Image
from typing import Tuple


class MaskPostProcessor:
    """Handles post-processing of model outputs to RGB masks."""
    
    def __init__(self, original_size: Tuple[int, int] = (540, 540)):
        """
        Initialize post-processor.
        
        Args:
            original_size: Target size for output masks (default: 540x540)
        """
        self.original_size = original_size
        
        # Define class to RGB color mappings
        self.class_to_color = {
            0: (0, 0, 0),       # Background/tools -> Black
            1: (255, 0, 0),     # Remaining tumor -> Red
            2: (0, 0, 255),     # Resected tumor -> Blue
            3: (0, 255, 0),     # Charred cut line -> Green
        }
    
    def postprocess_mask(self, logits: torch.Tensor) -> np.ndarray:
        """
        Convert model logits to RGB mask at original resolution.
        
        Steps:
        1. Apply argmax to get class indices (softmax already applied by model)
        2. Convert class indices to RGB colors
        3. Resize from 512x512 to 540x540
        
        Args:
            logits: Model output (1, 3, 512, 512) or (B, 3, 512, 512)
            
        Returns:
            RGB mask as numpy array (540, 540, 3)
        """
        # Remove batch dimension if present and move to CPU
        if logits.dim() == 4:
            logits = logits.squeeze(0)  # (3, 512, 512)
        
        # Convert to numpy and move to CPU if on GPU
        logits_np = logits.detach().cpu().numpy()
        
        # Apply argmax to get class indices (C, H, W) -> (H, W)
        class_mask = np.argmax(logits_np, axis=0)  # (512, 512)
        
        # Convert class indices to RGB colors
        rgb_mask = self._class_to_rgb(class_mask)  # (512, 512, 3)
        
        # Resize from 512x512 to original size (540x540)
        pil_mask = Image.fromarray(rgb_mask.astype(np.uint8))
        resized_mask = pil_mask.resize(self.original_size, Image.NEAREST)
        
        # Convert back to numpy array
        final_mask = np.array(resized_mask)
        
        return final_mask
    
    def apply_spatial_filters(
        self,
        rgb_mask: np.ndarray,
        cc_min_area: int = 0,
        morph_close_kernel: int = 0,
    ) -> np.ndarray:
        """
        Apply connected component filtering and/or morphological closing to an RGB mask.

        Order: closing first (merges nearby fragments), then CC filtering (removes small blobs).

        Args:
            rgb_mask: RGB mask (H, W, 3) with black/red/blue encoding
            cc_min_area: Remove connected components with area < this (pixels). 0 = disabled.
            morph_close_kernel: Ellipse kernel size for morphological closing. 0 = disabled.

        Returns:
            Filtered RGB mask (H, W, 3)
        """
        import cv2

        h, w = rgb_mask.shape[:2]

        # Convert RGB → integer class mask (0=bg, 1=remaining, 2=resected)
        class_mask = np.zeros((h, w), dtype=np.uint8)
        class_mask[(rgb_mask[:, :, 0] == 255) & (rgb_mask[:, :, 2] == 0)] = 1
        class_mask[(rgb_mask[:, :, 2] == 255) & (rgb_mask[:, :, 0] == 0)] = 2

        result = class_mask.copy()

        for cls in [1, 2]:
            binary = (class_mask == cls).astype(np.uint8)
            if binary.sum() == 0:
                continue

            # 1. Morphological closing — fills holes, smooths edges
            if morph_close_kernel > 0:
                k = morph_close_kernel | 1  # ensure odd
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # 2. Connected component filtering — removes small blobs
            if cc_min_area > 0:
                n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    binary, connectivity=8
                )
                filtered = np.zeros_like(binary)
                for lbl in range(1, n_labels):
                    if stats[lbl, cv2.CC_STAT_AREA] >= cc_min_area:
                        filtered[labels == lbl] = 1
                binary = filtered

            # Write back: clear original class pixels, paint filtered result
            result[class_mask == cls] = 0
            result[binary == 1] = cls

        # Convert back to RGB
        rgb_result = np.zeros_like(rgb_mask)
        rgb_result[result == 1] = [255, 0, 0]
        rgb_result[result == 2] = [0, 0, 255]
        return rgb_result

    def _class_to_rgb(self, class_mask: np.ndarray) -> np.ndarray:
        """
        Convert class indices to RGB colors.
        
        Mapping:
        - Class 0 -> Black (0, 0, 0)
        - Class 1 -> Red (255, 0, 0)
        - Class 2 -> Blue (0, 0, 255)
        
        Args:
            class_mask: Class indices (H, W)
            
        Returns:
            RGB mask (H, W, 3)
        """
        height, width = class_mask.shape
        rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Map each class to its corresponding RGB color
        for class_idx, color in self.class_to_color.items():
            mask = (class_mask == class_idx)
            rgb_mask[mask] = color
        
        return rgb_mask
