"""
Model persistence infrastructure for saving and loading trained models.

This module provides functionality for:
- Saving model weights in PyTorch .pth format
- Saving architecture configuration and hyperparameters in JSON format
- Computing and storing file checksums for integrity verification
- Loading models with checksum verification
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class ModelCheckpointSaver:
    """
    Handles saving model checkpoints with metadata and integrity verification.
    
    Saves:
    - Model state dict (.pth file)
    - Metadata JSON with architecture config, hyperparameters, preprocessing params
    - File checksums for integrity verification
    """
    
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        """
        Initialize checkpoint saver.
        
        Args:
            checkpoint_dir: Directory to save checkpoints (default: './checkpoints')
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        metadata: Dict[str, Any],
        checkpoint_name: str = 'model_checkpoint'
    ) -> Dict[str, str]:
        """
        Save model checkpoint with metadata and checksums.
        
        Args:
            model: PyTorch model to save
            metadata: Dictionary containing architecture config, hyperparameters, etc.
            checkpoint_name: Base name for checkpoint files (default: 'model_checkpoint')
            
        Returns:
            Dictionary with paths to saved files:
                - 'model_path': Path to .pth file
                - 'metadata_path': Path to metadata JSON file
                - 'model_checksum': SHA256 checksum of model file
                - 'metadata_checksum': SHA256 checksum of metadata file
                
        Example metadata structure:
            {
                "architecture": "UNet-ResNet34",
                "num_classes": 3,
                "input_size": [512, 512],
                "output_size": [540, 540],
                "pretrained_encoder": true,
                "training_config": {
                    "learning_rate": 0.001,
                    "optimizer": "Adam",
                    "loss_function": "CrossEntropy + Dice",
                    "num_epochs": 100,
                    "early_stopping_patience": 10,
                    "batch_size": 8
                },
                "preprocessing": {
                    "normalization": "min-max [0, 1]",
                    "resize_method": "bilinear",
                    "augmentation": {
                        "brightness_range": [0.8, 1.2],
                        "contrast_range": [0.8, 1.2]
                    }
                },
                "dataset_split": {
                    "train": 0.8,
                    "validation": 0.1,
                    "test": 0.1,
                    "random_seed": 42
                },
                "color_mapping": {
                    "class_0": [0, 0, 0],
                    "class_1": [255, 0, 0],
                    "class_2": [0, 0, 255]
                },
                "color_threshold": 30,
                "epoch": 50,
                "val_loss": 0.123,
                "dice_remaining": 0.85,
                "dice_resected": 0.82
            }
        """
        # Define file paths
        model_path = self.checkpoint_dir / f'{checkpoint_name}.pth'
        metadata_path = self.checkpoint_dir / f'{checkpoint_name}_metadata.json'
        
        # Save model state dict
        torch.save(model.state_dict(), model_path)
        
        # Compute model file checksum
        model_checksum = self._compute_checksum(model_path)
        
        # Add checksum to metadata
        metadata_with_checksum = metadata.copy()
        metadata_with_checksum['model_checksum'] = model_checksum
        
        # Save metadata as JSON
        with open(metadata_path, 'w') as f:
            json.dump(metadata_with_checksum, f, indent=2)
        
        # Compute metadata file checksum
        metadata_checksum = self._compute_checksum(metadata_path)
        
        return {
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'model_checksum': model_checksum,
            'metadata_checksum': metadata_checksum
        }
    
    def _compute_checksum(self, file_path: Path) -> str:
        """
        Compute SHA256 checksum of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal SHA256 checksum string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b''):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()



class ModelCheckpointLoader:
    """
    Handles loading model checkpoints with integrity verification.
    
    Loads:
    - Model state dict from .pth file
    - Metadata from JSON file
    - Verifies file integrity using checksums
    """
    
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        """
        Initialize checkpoint loader.
        
        Args:
            checkpoint_dir: Directory containing checkpoints (default: './checkpoints')
        """
        self.checkpoint_dir = Path(checkpoint_dir)
    
    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_name: str = 'model_checkpoint',
        device: str = 'cpu',
        verify_checksum: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint with integrity verification.
        
        Args:
            model: PyTorch model to load weights into
            checkpoint_name: Base name of checkpoint files (default: 'model_checkpoint')
            device: Device to load model to ('cpu' or 'cuda')
            verify_checksum: Whether to verify file integrity (default: True)
            
        Returns:
            Dictionary containing loaded metadata
            
        Raises:
            FileNotFoundError: If checkpoint files don't exist
            IntegrityError: If checksum verification fails
            RuntimeError: If model loading fails
        """
        # Define file paths
        model_path = self.checkpoint_dir / f'{checkpoint_name}.pth'
        metadata_path = self.checkpoint_dir / f'{checkpoint_name}_metadata.json'
        
        # Check if files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Verify checksums if requested
        if verify_checksum:
            self._verify_integrity(model_path, metadata_path, metadata)
        
        # Load model state dict
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")
        
        return metadata
    
    def _verify_integrity(
        self,
        model_path: Path,
        metadata_path: Path,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Verify file integrity using checksums.
        
        Args:
            model_path: Path to model file
            metadata_path: Path to metadata file
            metadata: Loaded metadata containing expected checksums
            
        Raises:
            IntegrityError: If checksum verification fails
        """
        # Verify model file checksum
        if 'model_checksum' in metadata:
            expected_checksum = metadata['model_checksum']
            actual_checksum = self._compute_checksum(model_path)
            
            if actual_checksum != expected_checksum:
                raise IntegrityError(
                    f"Model file checksum mismatch. "
                    f"Expected: {expected_checksum}, "
                    f"Actual: {actual_checksum}. "
                    f"File may be corrupted."
                )
    
    def _compute_checksum(self, file_path: Path) -> str:
        """
        Compute SHA256 checksum of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal SHA256 checksum string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b''):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


class IntegrityError(Exception):
    """Exception raised when file integrity verification fails."""
    pass
