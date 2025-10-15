"""
Training utilities and functions
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Optional, Dict
import numpy as np
from tqdm import tqdm


class Trainer:
    """Training class for PyTorch models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device: str = 'cuda',
        scheduler=None,
        save_dir: str = './models/checkpoints',
        experiment_name: str = 'experiment'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def fit(self, num_epochs: int, early_stopping_patience: int = 5):
        """Train the model for multiple epochs"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print("-" * 80)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            epoch_time = time.time() - start_time
            
            # Print metrics
            print(f"\nEpoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, is_best=True)
                patience_counter = 0
                print("✓ New best model saved!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            print("-" * 80)
        
        print("\nTraining completed!")
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if is_best:
            filepath = os.path.join(self.save_dir, f'{self.experiment_name}_best.pth')
        else:
            filepath = os.path.join(self.save_dir, f'{self.experiment_name}_epoch_{epoch}.pth')
        
        torch.save(checkpoint, filepath)


def get_optimizer(model, optimizer_name: str = 'adam', lr: float = 0.001, **kwargs):
    """
    Get optimizer by name
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw')
        lr: Learning rate
        **kwargs: Additional optimizer arguments
    """
    optimizers = {
        'adam': Adam,
        'sgd': SGD,
        'adamw': AdamW,
    }
    
    if optimizer_name not in optimizers:
        raise ValueError(f"Optimizer {optimizer_name} not found")
    
    return optimizers[optimizer_name](model.parameters(), lr=lr, **kwargs)


def get_scheduler(optimizer, scheduler_name: str = 'reduce_on_plateau', **kwargs):
    """
    Get learning rate scheduler by name
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Name of scheduler
        **kwargs: Scheduler-specific arguments
    """
    schedulers = {
        'reduce_on_plateau': ReduceLROnPlateau,
        'cosine': CosineAnnealingLR,
    }
    
    if scheduler_name not in schedulers:
        raise ValueError(f"Scheduler {scheduler_name} not found")
    
    return schedulers[scheduler_name](optimizer, **kwargs)
