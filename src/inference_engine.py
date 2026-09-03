"""
Inference engine for generating segmentation masks using trained models.

This module provides functionality for:
- Loading trained models with metadata
- Generating segmentation masks for new images
- Batch processing multiple images
- Uploading results to Google Drive
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from PIL import Image

from .model import UNetSegmentationModel
from .model_persistence import ModelCheckpointLoader
from .preprocessor import ImagePreprocessor
from .postprocessor import MaskPostProcessor


class InferenceEngine:
    """
    Inference engine for generating tumor segmentation masks.
    
    Loads a trained U-Net model and applies it to new images to generate
    segmentation masks identifying remaining and resected tumor regions.
    """
    
    def __init__(
        self,
        model_path: str,
        metadata_path: str,
        device: str = 'cuda'
    ):
        """
        Initialize inference engine.
        
        Loads the trained model and metadata from disk, verifies integrity,
        and prepares the model for inference on the specified device.
        
        Args:
            model_path: Path to saved model weights (.pth file)
            metadata_path: Path to model metadata (JSON file)
            device: Device for inference ('cuda' or 'cpu', default: 'cuda')
                   Automatically falls back to 'cpu' if CUDA is not available
        
        Raises:
            FileNotFoundError: If model or metadata files don't exist
            IntegrityError: If checksum verification fails
            RuntimeError: If model loading fails
        """
        # Validate device availability
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = 'cpu'
        
        self.device = device
        
        # Extract checkpoint directory and name from paths
        model_path_obj = Path(model_path)
        checkpoint_dir = str(model_path_obj.parent)
        checkpoint_name = model_path_obj.stem  # Remove .pth extension
        
        # Initialize model checkpoint loader
        self.loader = ModelCheckpointLoader(checkpoint_dir=checkpoint_dir)
        
        # Load metadata first to get architecture information
        import json
        metadata_path_obj = Path(metadata_path)
        if not metadata_path_obj.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path_obj, 'r') as f:
            metadata_dict = json.load(f)
        
        # Parse architecture from metadata (format: "UNet-ResNet50" or "UNet++-EfficientNet-B2")
        architecture_str = metadata_dict.get('architecture', 'UNet-ResNet34')
        
        # Extract architecture type and encoder name
        if architecture_str.startswith('UNet++'):
            architecture = 'unetplusplus'
            encoder_part = architecture_str.replace('UNet++-', '')
        elif architecture_str.startswith('UNet'):
            architecture = 'unet'
            encoder_part = architecture_str.replace('UNet-', '')
        else:
            # Fallback to defaults
            architecture = 'unet'
            encoder_part = 'ResNet34'
        
        # Normalize encoder name (ResNet50 -> resnet50, EfficientNet-B2 -> efficientnet-b2)
        encoder_name = encoder_part.lower().replace('resnet', 'resnet').replace('efficientnet-b', 'efficientnet-b')
        
        # Create model instance with correct architecture and encoder
        # num_classes defaults to 3 for backward compatibility with old checkpoints
        num_classes = metadata_dict.get('num_classes', 3)
        self.model = UNetSegmentationModel(
            num_classes=num_classes,
            pretrained=False,
            architecture=architecture,
            encoder_name=encoder_name
        )
        
        # Load model weights and metadata with error handling
        try:
            self.metadata = self.loader.load_checkpoint(
                model=self.model,
                checkpoint_name=checkpoint_name,
                device=self.device,
                verify_checksum=True
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model loading failed: {e}. "
                f"Please ensure the checkpoint files exist at {checkpoint_dir}"
            )
        except Exception as e:
            # Catch IntegrityError and other loading failures
            raise RuntimeError(
                f"Model loading failed: {e}. "
                f"The checkpoint file may be corrupted or incompatible. "
                f"Please verify the model file or retrain the model."
            )
        
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        
        # Initialize preprocessor with parameters from metadata
        input_size = tuple(self.metadata.get('input_size', [512, 512]))
        self.preprocessor = ImagePreprocessor(target_size=input_size)
        
        # Initialize post-processor with parameters from metadata
        output_size = tuple(self.metadata.get('output_size', [540, 540]))
        self.postprocessor = MaskPostProcessor(original_size=output_size)
    
    def segment_image(self, image_path: str) -> np.ndarray:
        """
        Generate segmentation mask for a single image.
        
        Args:
            image_path: Path to input image file
            
        Returns:
            RGB mask as numpy array with shape (H, W, 3) where:
            - Black (0, 0, 0): Background/tools (class 0)
            - Red (255, 0, 0): Remaining tumor (class 1)
            - Blue (0, 0, 255): Resected tumor (class 2)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is not supported or invalid
            RuntimeError: If inference fails
        """
        # Load image
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and preprocess image with error handling
        try:
            image = Image.open(image_path_obj).convert('RGB')
            image_array = np.array(image)
            
            # Validate image dimensions
            if image_array.ndim != 3 or image_array.shape[2] != 3:
                raise ValueError(
                    f"Invalid image format: expected RGB image with 3 channels, "
                    f"got shape {image_array.shape}"
                )
            
        except (IOError, OSError) as e:
            raise ValueError(
                f"Unable to load image {image_path}: {e}. "
                f"The file may be corrupted or in an unsupported format."
            )
        
        # Preprocess image
        try:
            preprocessed = self.preprocessor.preprocess_image(image_array)
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed for {image_path}: {e}")
        
        # Add batch dimension and move to device
        input_tensor = preprocessed.unsqueeze(0).to(self.device)
        
        # Run inference
        try:
            with torch.no_grad():
                output = self.model(input_tensor)
        except Exception as e:
            raise RuntimeError(f"Inference failed for {image_path}: {e}")
        
        # Post-process output to RGB mask
        try:
            rgb_mask = self.postprocessor.postprocess_mask(output)
        except Exception as e:
            raise RuntimeError(f"Post-processing failed for {image_path}: {e}")
        
        return rgb_mask

    def predict_single(
        self,
        image: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate probabilities and preprocessed frame for single image.

        This method is designed for video inference with EMA smoothing, where
        both the probability map and preprocessed frame are needed for temporal
        blending and scene change detection.

        Args:
            image: Input RGB image as numpy array [H, W, 3]

        Returns:
            Tuple of:
            - probs: Probability tensor [C, H, W] on device, where C is number of classes
                    Values are in [0, 1] (softmax output)
            - frame: Preprocessed normalized frame [C, H, W] on device
                    Values are in [0, 1] (normalized)

        Raises:
            ValueError: If image format is invalid
            RuntimeError: If preprocessing or inference fails

        Note:
            For simple inference without EMA smoothing, use segment_image() instead.
        """
        # Validate image dimensions
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Invalid image format: expected RGB image with 3 channels, "
                f"got shape {image.shape}"
            )

        # Preprocess image
        try:
            preprocessed = self.preprocessor.preprocess_image(image)
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {e}")

        # Add batch dimension and move to device
        input_tensor = preprocessed.unsqueeze(0).to(self.device)

        # Run inference to get probabilities
        try:
            with torch.no_grad():
                output = self.model(input_tensor)
                # Apply softmax to get probabilities [1, C, H, W]
                probs = torch.softmax(output, dim=1)
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

        # Remove batch dimension: [1, C, H, W] -> [C, H, W]
        probs = probs.squeeze(0)
        preprocessed_frame = preprocessed.to(self.device)

        return probs, preprocessed_frame

    def predict_with_tta(
        self,
        image: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TTA inference: average original + horizontal-flip predictions.

        Adds ~1% Dice at 2× the single-pass cost (still real-time viable
        at ~25 FPS if the base model runs ~50 FPS).

        Args:
            image: Input RGB image as numpy array [H, W, 3]

        Returns:
            Same as predict_single: (probs [C, H, W], frame [C, H, W])
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB [H,W,3], got {image.shape}")

        preprocessed = self.preprocessor.preprocess_image(image)
        x = preprocessed.unsqueeze(0).to(self.device)  # (1, 3, H, W)

        with torch.no_grad():
            # Original
            probs_orig = torch.softmax(self.model(x), dim=1)  # (1, C, H, W)

            # Horizontal flip + unflip output
            x_flip = torch.flip(x, dims=[3])
            probs_flip = torch.softmax(self.model(x_flip), dim=1)
            probs_flip = torch.flip(probs_flip, dims=[3])  # unflip

        probs = ((probs_orig + probs_flip) * 0.5).squeeze(0)  # (C, H, W)
        return probs, preprocessed.to(self.device)

    def batch_segment(
        self,
        image_paths: List[str],
        output_dir: str,
        drive_service: Optional[Any] = None,
        drive_output_folder_id: str = '1uZtGGpgI4GkDOzfKg9bRYoFWVDdUgSTb'
    ) -> List[str]:
        """
        Segment multiple images and save results to disk and optionally Google Drive.
        
        Args:
            image_paths: List of input image file paths
            output_dir: Local directory for saving masks
            drive_service: Optional authenticated Google Drive service for upload
            drive_output_folder_id: Google Drive folder ID for output
                                   (default: 1uZtGGpgI4GkDOzfKg9bRYoFWVDdUgSTb)
            
        Returns:
            List of output mask file paths (local paths) for successfully processed images
            
        Note:
            Invalid images are skipped and logged. The method continues processing
            remaining images even if some fail.
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_mask_paths = []
        skipped_images = []
        
        for image_path in image_paths:
            try:
                # Generate segmentation mask
                rgb_mask = self.segment_image(image_path)
                
                # Get original filename without extension
                image_path_obj = Path(image_path)
                base_filename = image_path_obj.stem  # e.g., 'frame000211' from 'frame000211.png'
                
                # Generate output filenames preserving original name
                mask_filename = f'{base_filename}_mask.png'
                org_filename = f'{base_filename}_org.png'
                
                # Save mask locally
                mask_output_path = output_path / mask_filename
                mask_image = Image.fromarray(rgb_mask.astype(np.uint8))
                mask_image.save(mask_output_path)
                output_mask_paths.append(str(mask_output_path))
                
                # Copy original image to output directory
                org_output_path = output_path / org_filename
                original_image = Image.open(image_path)
                original_image.save(org_output_path)
                
                # Upload to Google Drive if service is provided
                if drive_service is not None:
                    # Upload mask with retry logic
                    try:
                        self._upload_to_drive_with_retry(
                            drive_service,
                            str(mask_output_path),
                            drive_output_folder_id
                        )
                    except RuntimeError as e:
                        print(f"Warning: Failed to upload mask after retries: {e}")
                        print(f"Mask saved locally at: {mask_output_path}")
                    
                    # Upload original image with retry logic
                    try:
                        self._upload_to_drive_with_retry(
                            drive_service,
                            str(org_output_path),
                            drive_output_folder_id
                        )
                    except RuntimeError as e:
                        print(f"Warning: Failed to upload original image after retries: {e}")
                        print(f"Original image saved locally at: {org_output_path}")
                
            except (FileNotFoundError, ValueError) as e:
                # Skip invalid images and continue with batch
                print(f"Skipping invalid image {image_path}: {e}")
                skipped_images.append(image_path)
                continue
            except Exception as e:
                # Log unexpected errors and continue
                print(f"Error processing {image_path}: {e}")
                skipped_images.append(image_path)
                continue
        
        # Log summary
        if skipped_images:
            print(f"\nBatch processing complete. Skipped {len(skipped_images)} invalid images:")
            for img in skipped_images:
                print(f"  - {img}")
        
        return output_mask_paths
    
    def _extract_frame_number(self, filename: str) -> str:
        """
        Extract 4-digit frame number from filename.
        
        Args:
            filename: Image filename (e.g., 'frame_0123_org.png')
            
        Returns:
            4-digit frame number as string (e.g., '0123')
            
        Raises:
            ValueError: If frame number cannot be extracted
        """
        import re
        
        # Pattern to match 4-digit frame number
        pattern = r'(\d{4})'
        match = re.search(pattern, filename)
        
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Could not extract frame number from filename: {filename}")
    
    def _upload_to_drive(
        self,
        drive_service: Any,
        file_path: str,
        folder_id: str
    ) -> None:
        """
        Upload a file to Google Drive.
        
        Args:
            drive_service: Authenticated Google Drive service
            file_path: Path to local file to upload
            folder_id: Google Drive folder ID for upload destination
            
        Raises:
            RuntimeError: If upload fails
        """
        try:
            from googleapiclient.http import MediaFileUpload
            
            file_metadata = {
                'name': Path(file_path).name,
                'parents': [folder_id]
            }
            
            media = MediaFileUpload(file_path, resumable=True)
            
            drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
        except Exception as e:
            raise RuntimeError(f"Failed to upload {file_path} to Google Drive: {e}")
    
    def _upload_to_drive_with_retry(
        self,
        drive_service: Any,
        file_path: str,
        folder_id: str,
        max_retries: int = 3
    ) -> None:
        """
        Upload a file to Google Drive with retry logic.
        
        Attempts to upload the file up to max_retries times. If all attempts fail,
        the file remains saved locally and a RuntimeError is raised.
        
        Args:
            drive_service: Authenticated Google Drive service
            file_path: Path to local file to upload
            folder_id: Google Drive folder ID for upload destination
            max_retries: Maximum number of upload attempts (default: 3)
            
        Raises:
            RuntimeError: If all upload attempts fail
        """
        import time
        
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                self._upload_to_drive(drive_service, file_path, folder_id)
                # Upload succeeded
                return
            except RuntimeError as e:
                last_error = e
                if attempt < max_retries:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** attempt  # 2, 4, 8 seconds
                    print(f"Upload attempt {attempt} failed. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # All retries exhausted
                    print(f"Upload failed after {max_retries} attempts.")
        
        # All retries failed
        raise RuntimeError(
            f"Failed to upload {file_path} after {max_retries} attempts. "
            f"Last error: {last_error}. File saved locally at: {file_path}"
        )
