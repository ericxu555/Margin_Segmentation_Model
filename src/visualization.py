"""Visualization module for generating overlay images of segmentation results."""

import numpy as np
from PIL import Image
from typing import Tuple


class VisualizationGenerator:
    """Generates overlay visualizations of segmentation masks on original images."""
    
    def generate_overlay(self, original_image: np.ndarray, mask: np.ndarray,
                        alpha: float = 0.5) -> np.ndarray:
        """
        Create overlay visualization combining original image and segmentation mask.
        
        Overlays the predicted mask on the original image with specified transparency.
        - Remaining tumor (red pixels in mask) rendered with red overlay
        - Resected tumor (blue pixels in mask) rendered with blue overlay
        - Background/tools (black pixels in mask) not overlaid
        
        Args:
            original_image: Original image as numpy array (H, W, 3) with values in [0, 255]
            mask: RGB segmentation mask as numpy array (H, W, 3) with values in [0, 255]
                  Red (255, 0, 0) = remaining tumor
                  Blue (0, 0, 255) = resected tumor
                  Black (0, 0, 0) = background/tools
            alpha: Transparency for overlay (0.5 = 50% transparency)
            
        Returns:
            Overlay image as numpy array (H, W, 3) with values in [0, 255]
        """
        # Ensure inputs are numpy arrays with correct dtype
        if not isinstance(original_image, np.ndarray):
            original_image = np.array(original_image)
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        # Ensure images have the same dimensions
        if original_image.shape != mask.shape:
            raise ValueError(
                f"Image and mask dimensions must match. "
                f"Got image: {original_image.shape}, mask: {mask.shape}"
            )
        
        # Convert to float for blending calculations
        original_float = original_image.astype(np.float32)
        mask_float = mask.astype(np.float32)
        
        # Create overlay by blending original image with mask
        # Start with the original image
        overlay = original_float.copy()
        
        # Identify tumor regions (non-black pixels in mask)
        # Red tumor: pixels where R=255, G=0, B=0
        red_tumor_mask = (mask[:, :, 0] == 255) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)
        
        # Blue tumor: pixels where R=0, G=0, B=255
        blue_tumor_mask = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255)
        
        # Apply alpha blending for red tumor regions
        # Formula: output = (1 - alpha) * original + alpha * mask
        overlay[red_tumor_mask] = (1 - alpha) * original_float[red_tumor_mask] + alpha * mask_float[red_tumor_mask]
        
        # Apply alpha blending for blue tumor regions
        overlay[blue_tumor_mask] = (1 - alpha) * original_float[blue_tumor_mask] + alpha * mask_float[blue_tumor_mask]
        
        # Convert back to uint8
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay
