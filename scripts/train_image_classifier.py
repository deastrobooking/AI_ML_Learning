"""
Example training script for image classification
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import random_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.data_utils import CustomImageDataset, get_image_transforms, create_data_loaders
from src.training import Trainer, get_optimizer, get_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--data_dir', type=str, default='./data/raw', help='Path to data directory')
    parser.add_argument('--model', type=str, default='simple_cnn', choices=['simple_cnn', 'resnet'], help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'], help='Optimizer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--save_dir', type=str, default='./models/checkpoints', help='Directory to save models')
    parser.add_argument('--experiment_name', type=str, default='image_classification', help='Experiment name')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("Image Classification Training")
    print("=" * 80)
    print(f"Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 80)
    
    # Set device
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # Create transforms
    train_transform = get_image_transforms(image_size=224, is_training=True)
    val_transform = get_image_transforms(image_size=224, is_training=False)
    
    # Load dataset
    print("\nLoading dataset...")
    full_dataset = CustomImageDataset(args.data_dir, transform=train_transform)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update validation transform
    val_dataset.dataset.transform = val_transform
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, 
        val_dataset, 
        batch_size=args.batch_size
    )
    
    # Create model
    print(f"\nCreating model: {args.model}")
    model = get_model(args.model, num_classes=args.num_classes)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args.optimizer, lr=args.lr)
    scheduler = get_scheduler(optimizer, 'reduce_on_plateau', patience=3, factor=0.5)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        scheduler=scheduler,
        save_dir=args.save_dir,
        experiment_name=args.experiment_name
    )
    
    # Train
    history = trainer.fit(num_epochs=args.epochs, early_stopping_patience=5)
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Best validation loss: {min(history['val_losses']):.4f}")
    print(f"Best validation accuracy: {max(history['val_accuracies']):.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()
