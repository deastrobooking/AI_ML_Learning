"""
Utility functions for data loading and preprocessing
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple, List


class CustomImageDataset(Dataset):
    """Custom Dataset for loading images"""
    
    def __init__(self, image_dir: str, labels: Optional[List] = None, transform=None):
        """
        Args:
            image_dir: Directory containing images
            labels: Optional list of labels
            transform: Optional transform to be applied on images
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        return image


def get_image_transforms(image_size: int = 224, is_training: bool = True):
    """
    Get image transformation pipeline
    
    Args:
        image_size: Target image size
        is_training: Whether this is for training (includes augmentation)
        
    Returns:
        torchvision.transforms.Compose object
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def load_csv_data(file_path: str, target_col: Optional[str] = None) -> Tuple:
    """
    Load data from CSV file
    
    Args:
        file_path: Path to CSV file
        target_col: Name of target column (optional)
        
    Returns:
        Tuple of (features, targets) or just features if target_col is None
    """
    df = pd.read_csv(file_path)
    
    if target_col:
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        return X, y
    
    return df.values


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath} (Epoch {epoch}, Loss: {loss:.4f})")
    return epoch, loss
