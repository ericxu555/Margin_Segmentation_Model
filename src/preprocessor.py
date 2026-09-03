"""Image and mask preprocessing module."""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Exception raised for preprocessing errors."""
    pass


class ImagePreprocessor:
    """Handles image and mask preprocessing for model input."""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image dimensions for model input
        """
        self.target_size = target_size
        
        # Define color mappings for mask conversion
        self.color_to_class = {
            'red': (255, 0, 0, 1),      # Remaining tumor -> class 1
            'blue': (0, 0, 255, 2),     # Resected tumor -> class 2
            'background': (0, 0, 0, 0)  # Background/tools -> class 0
        }
        self.distance_threshold = 80
        
        # Track ambiguous pixels for logging
        self.ambiguous_pixel_count = 0
        self.total_pixel_count = 0
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Resize and normalize image.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            
        Returns:
            Preprocessed tensor (3, 512, 512) with values in [0, 1]
            
        Raises:
            PreprocessingError: If image dimensions are invalid or aspect ratio is severely distorted
        """
        # Validate input
        if not isinstance(image, np.ndarray) and not isinstance(image, Image.Image):
            raise PreprocessingError(f"Invalid image type: {type(image)}. Expected numpy array or PIL Image.")
        
        # Convert to PIL Image for resizing
        if isinstance(image, np.ndarray):
            # Validate dimensions
            if image.ndim != 3 or image.shape[2] != 3:
                raise PreprocessingError(
                    f"Invalid image dimensions: {image.shape}. Expected (H, W, 3)."
                )
            
            original_height, original_width = image.shape[:2]
            
            # Check for severely distorted aspect ratio (more than 2:1 or 1:2)
            aspect_ratio = original_width / original_height
            if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                logger.warning(
                    f"Image has unusual aspect ratio {aspect_ratio:.2f} "
                    f"(dimensions: {original_width}x{original_height}). "
                    f"Resizing may cause distortion."
                )
            
            # Handle different input formats
            if image.dtype == np.uint8:
                pil_image = Image.fromarray(image)
            else:
                # If already float, convert to uint8 first
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            pil_image = image
            original_width, original_height = pil_image.size
        
        # Log if dimensions don't match expected input size
        if (original_width, original_height) != (540, 540):
            logger.info(
                f"Image dimensions {original_width}x{original_height} differ from expected 540x540. "
                f"Resizing to {self.target_size[0]}x{self.target_size[1]}."
            )
        
        try:
            # Resize using bilinear interpolation
            resized_image = pil_image.resize(self.target_size, Image.BILINEAR)
        except Exception as e:
            raise PreprocessingError(f"Failed to resize image: {str(e)}")
        
        # Convert to numpy array
        image_array = np.array(resized_image)
        
        # Normalize to [0, 1]
        normalized_image = image_array.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(normalized_image).permute(2, 0, 1)
        
        return tensor
    
    def preprocess_mask(self, mask: np.ndarray, random_seed: int = 42) -> torch.Tensor:
        """
        Convert RGB mask to class indices.
        
        Color mapping:
        - Red (255, 0, 0) -> class 1 (remaining tumor)
        - Blue (0, 0, 255) -> class 2 (resected tumor)
        - Other colors -> class 0 (background/tools)
        
        Uses nearest-neighbor with distance threshold of 30.
        Ambiguous pixels assigned to class 0.
        
        Args:
            mask: RGB mask as numpy array (H, W, 3)
            random_seed: Random seed for reproducible dataset splitting (default: 42)
            
        Returns:
            Class index tensor (512, 512) with values in {0, 1, 2}
            
        Raises:
            PreprocessingError: If mask dimensions are invalid
        """
        # Validate input
        if not isinstance(mask, np.ndarray) and not isinstance(mask, Image.Image):
            raise PreprocessingError(f"Invalid mask type: {type(mask)}. Expected numpy array or PIL Image.")
        
        # Convert to PIL Image for resizing
        if isinstance(mask, np.ndarray):
            # Validate dimensions
            if mask.ndim != 3 or mask.shape[2] != 3:
                raise PreprocessingError(
                    f"Invalid mask dimensions: {mask.shape}. Expected (H, W, 3)."
                )
            
            if mask.dtype == np.uint8:
                pil_mask = Image.fromarray(mask)
            else:
                pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
        else:
            pil_mask = mask
        
        try:
            # Resize using nearest neighbor to preserve color values
            resized_mask = pil_mask.resize(self.target_size, Image.NEAREST)
        except Exception as e:
            raise PreprocessingError(f"Failed to resize mask: {str(e)}")
        
        # Convert to numpy array
        mask_array = np.array(resized_mask)
        
        # Reset ambiguous pixel tracking for this mask
        self.ambiguous_pixel_count = 0
        self.total_pixel_count = self.target_size[0] * self.target_size[1]
        
        # Convert RGB to class indices
        class_mask = np.zeros((self.target_size[0], self.target_size[1]), dtype=np.int64)
        
        for i in range(mask_array.shape[0]):
            for j in range(mask_array.shape[1]):
                rgb_pixel = mask_array[i, j]
                class_idx, is_ambiguous = self._rgb_to_class(rgb_pixel)
                class_mask[i, j] = class_idx
                if is_ambiguous:
                    self.ambiguous_pixel_count += 1
        
        # Log warning if significant percentage of pixels are ambiguous
        if self.total_pixel_count > 0:
            ambiguous_percentage = (self.ambiguous_pixel_count / self.total_pixel_count) * 100
            if ambiguous_percentage > 5.0:  # Warn if more than 5% are ambiguous
                logger.warning(
                    f"Mask contains {ambiguous_percentage:.2f}% ambiguous pixels "
                    f"({self.ambiguous_pixel_count}/{self.total_pixel_count}) that don't match "
                    f"expected colors within threshold {self.distance_threshold}. "
                    f"These pixels have been assigned to class 0 (background)."
                )
        
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(class_mask)
        
        return tensor
    
    def _rgb_to_class(self, rgb_pixel: np.ndarray) -> Tuple[int, bool]:
        """
        Convert RGB pixel to class index using nearest-neighbor.
        
        Args:
            rgb_pixel: RGB values as array [R, G, B]
            
        Returns:
            Tuple of (class_index, is_ambiguous)
            - class_index: Class index (0, 1, or 2)
            - is_ambiguous: True if pixel doesn't match expected colors within threshold
        """
        # Ensure pixel is numpy array and extract RGB values
        pixel_color = np.array(rgb_pixel[:3], dtype=np.float32)
        
        # Define reference colors
        red_color = np.array([255.0, 0.0, 0.0])
        blue_color = np.array([0.0, 0.0, 255.0])
        white_color = np.array([255.0, 255.0, 255.0])
        
        # Calculate Euclidean distances to all reference colors
        distance_to_red = np.linalg.norm(pixel_color - red_color)
        distance_to_blue = np.linalg.norm(pixel_color - blue_color)
        distance_to_white = np.linalg.norm(pixel_color - white_color)
        
        green_color = np.array([0.0, 255.0, 0.0])
        distance_to_green = np.linalg.norm(pixel_color - green_color)

        # Check if pixel is close to white (background)
        if distance_to_white <= self.distance_threshold:
            return 0, False  # White -> class 0 (background)

        distances = {
            1: distance_to_red,
            2: distance_to_blue,
            3: distance_to_green,
        }
        nearest_class = min(distances, key=distances.get)
        min_distance = distances[nearest_class]

        if min_distance > self.distance_threshold:
            return 0, True  # Too far from any known color -> background

        # Ambiguous if two classes are equally close
        sorted_dists = sorted(distances.values())
        if abs(sorted_dists[0] - sorted_dists[1]) < 1e-6:
            return 0, True

        return nearest_class, False


class AugmentationEngine:
    """Handles data augmentation during training with geometric and photometric transforms."""

    def __init__(self, 
                     brightness_range: Tuple[float, float] = (0.7, 1.3),
                     contrast_range: Tuple[float, float] = (0.8, 1.2),
                     hue_shift: float = 0.00,
                     saturation_range: Tuple[float, float] = (1.0, 1.0),
                     enable_geometric: bool = True,
                     flip_prob: float = 0.5,
                     rotation_prob: float = 0.5,
                     rotation_range: Tuple[float, float] = (-15.0, 15.0)):
            """
            Initialize augmentation engine.

            Args:
                brightness_range: Min and max brightness multipliers (wider range for robustness)
                contrast_range: Min and max contrast multipliers
                hue_shift: Maximum hue shift in both directions (±hue_shift)
                saturation_range: Min and max saturation multipliers
                enable_geometric: Enable geometric augmentations (flips, rotations)
                flip_prob: Probability of applying horizontal flip
                rotation_prob: Probability of applying rotation
                rotation_range: Min and max rotation angles in degrees
            """
            self.brightness_range = brightness_range
            self.contrast_range = contrast_range
            self.hue_shift = hue_shift
            self.saturation_range = saturation_range
            self.enable_geometric = enable_geometric
            self.flip_prob = flip_prob
            self.rotation_prob = rotation_prob
            self.rotation_range = rotation_range


    def augment(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random augmentation to image and mask.

        Args:
            image: Preprocessed image tensor (C, H, W) with values in [0, 1]
            mask: Preprocessed mask tensor (H, W) with class indices

        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
        # Apply geometric augmentations (affects both image and mask)
        if self.enable_geometric:
            # Horizontal flip
            if np.random.random() < self.flip_prob:
                image = torch.flip(image, dims=[2])  # Flip width dimension
                mask = torch.flip(mask, dims=[1])    # Flip width dimension

            # Rotation
            if np.random.random() < self.rotation_prob:
                angle = np.random.uniform(self.rotation_range[0], self.rotation_range[1])
                image, mask = self._rotate(image, mask, angle)

        # Apply random brightness adjustment (image only)
        brightness_factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        augmented_image = image * brightness_factor

        # Apply random contrast adjustment (image only)
        # Contrast adjustment: (image - mean) * contrast_factor + mean
        contrast_factor = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
        mean = augmented_image.mean()
        augmented_image = (augmented_image - mean) * contrast_factor + mean

        # Clip values to [0, 1] range
        augmented_image = torch.clamp(augmented_image, 0.0, 1.0)

        # Surgical-specific augmentations (image only, each independent)
        if np.random.random() < 0.3:
            augmented_image = self._apply_smoke(augmented_image)
        if np.random.random() < 0.3:
            augmented_image = self._apply_specular(augmented_image)
        if np.random.random() < 0.3:
            augmented_image = self._apply_blood_tint(augmented_image)

        augmented_image = torch.clamp(augmented_image, 0.0, 1.0)

        return augmented_image, mask

    def _apply_smoke(self, image: torch.Tensor) -> torch.Tensor:
        """Simulate surgical smoke: Gaussian blur + brightness reduction on a random region."""
        img_np = image.numpy().transpose(1, 2, 0)
        H, W = img_np.shape[:2]

        rh = np.random.randint(int(H * 0.10), int(H * 0.30))
        rw = np.random.randint(int(W * 0.10), int(W * 0.30))
        y  = np.random.randint(0, H - rh)
        x  = np.random.randint(0, W - rw)

        ks         = int(np.random.choice([15, 17, 19, 21, 23, 25]))
        sigma      = np.random.uniform(3.0, 8.0)
        brightness = np.random.uniform(0.5, 0.8)

        region  = img_np[y:y+rh, x:x+rw].copy()
        blurred = cv2.GaussianBlur(region, (ks, ks), sigmaX=sigma)
        img_np[y:y+rh, x:x+rw] = blurred * brightness

        return torch.from_numpy(img_np.transpose(2, 0, 1))

    def _apply_specular(self, image: torch.Tensor) -> torch.Tensor:
        """Simulate specular highlights: 1-3 bright circular patches."""
        img_np = image.numpy().transpose(1, 2, 0).copy()
        H, W = img_np.shape[:2]

        n_spots = np.random.randint(1, 4)
        for _ in range(n_spots):
            cx        = np.random.randint(0, W)
            cy        = np.random.randint(0, H)
            radius    = np.random.randint(5, 21)
            intensity = np.random.uniform(0.78, 1.0)  # 200-255 in [0,1]

            ys, xs = np.ogrid[:H, :W]
            circle = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
            img_np[circle] = np.clip(img_np[circle] + intensity * 0.5, 0.0, 1.0)

        return torch.from_numpy(img_np.transpose(2, 0, 1))

    def _apply_blood_tint(self, image: torch.Tensor) -> torch.Tensor:
        """Simulate blood/fluid: red channel boost on a small random region."""
        img_np = image.numpy().transpose(1, 2, 0).copy()
        H, W = img_np.shape[:2]

        rh = np.random.randint(int(H * 0.05), int(H * 0.15))
        rw = np.random.randint(int(W * 0.05), int(W * 0.15))
        y  = np.random.randint(0, H - rh)
        x  = np.random.randint(0, W - rw)

        boost = np.random.uniform(20.0 / 255.0, 50.0 / 255.0)
        img_np[y:y+rh, x:x+rw, 0] = np.clip(img_np[y:y+rh, x:x+rw, 0] + boost, 0.0, 1.0)

        return torch.from_numpy(img_np.transpose(2, 0, 1))
    
    def _rotate(self, image: torch.Tensor, mask: torch.Tensor, angle: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rotate image and mask by the given angle.
        
        Args:
            image: Image tensor (C, H, W)
            mask: Mask tensor (H, W)
            angle: Rotation angle in degrees
            
        Returns:
            Tuple of (rotated_image, rotated_mask)
        """
        import torch.nn.functional as F
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        
        # Create rotation matrix
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        
        # Affine transformation matrix for rotation around center
        # Format: [[cos, -sin, 0], [sin, cos, 0]]
        theta = torch.tensor([
            [cos_val, -sin_val, 0],
            [sin_val, cos_val, 0]
        ], dtype=torch.float32)
        
        # Add batch dimension for grid_sample
        image_batch = image.unsqueeze(0)  # (1, C, H, W)
        mask_batch = mask.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        
        # Create affine grid
        grid = F.affine_grid(theta.unsqueeze(0), image_batch.size(), align_corners=False)
        
        # Apply rotation to image (bilinear interpolation)
        rotated_image = F.grid_sample(image_batch, grid, mode='bilinear', 
                                      padding_mode='zeros', align_corners=False)
        
        # Apply rotation to mask (nearest neighbor to preserve class indices)
        rotated_mask = F.grid_sample(mask_batch, grid, mode='nearest', 
                                     padding_mode='zeros', align_corners=False)
        
        # Remove batch dimension
        rotated_image = rotated_image.squeeze(0)
        rotated_mask = rotated_mask.squeeze(0).squeeze(0).long()
        
        return rotated_image, rotated_mask



def split_dataset(image_mask_pairs: list, 
                  train_ratio: float = 0.8, 
                  val_ratio: float = 0.1, 
                  test_ratio: float = 0.1,
                  random_seed: int = 42) -> dict:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        image_mask_pairs: List of (image_path, mask_path) tuples
        train_ratio: Proportion of data for training (default: 0.8)
        val_ratio: Proportion of data for validation (default: 0.1)
        test_ratio: Proportion of data for testing (default: 0.1)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Dictionary with keys 'train', 'validation', 'test' containing lists of pairs
    """
    # Validate ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle the dataset
    shuffled_pairs = image_mask_pairs.copy()
    np.random.shuffle(shuffled_pairs)
    
    # Calculate split indices
    total_samples = len(shuffled_pairs)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    # Split the dataset
    train_set = shuffled_pairs[:train_end]
    val_set = shuffled_pairs[train_end:val_end]
    test_set = shuffled_pairs[val_end:]
    
    return {
        'train': train_set,
        'validation': val_set,
        'test': test_set
    }
