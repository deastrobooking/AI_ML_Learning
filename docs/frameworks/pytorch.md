# PyTorch Deep Dive

A comprehensive guide to PyTorch, one of the most popular deep learning frameworks.

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Tensors](#tensors)
4. [Autograd](#autograd)
5. [Building Neural Networks](#building-neural-networks)
6. [Training Loop](#training-loop)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)
9. [Common Patterns](#common-patterns)
10. [Resources](#resources)

---

## Introduction

PyTorch is an open-source machine learning framework developed by Meta (Facebook). It's known for its:
- **Dynamic computation graphs**: Build and modify networks on-the-fly
- **Pythonic design**: Intuitive and easy to learn
- **Strong GPU acceleration**: Seamless CUDA integration
- **Rich ecosystem**: Large community and extensive libraries

### Installation

```bash
# CPU version
pip install torch torchvision torchaudio

# CUDA 11.8 (check your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Core Concepts

### 1. Tensors

Tensors are the fundamental data structure in PyTorch - multi-dimensional arrays similar to NumPy arrays but with GPU support.

```python
import torch

# Creating tensors
x = torch.tensor([1, 2, 3])
y = torch.zeros(3, 4)
z = torch.randn(2, 3)  # Random normal distribution

# Move to GPU
if torch.cuda.is_available():
    x = x.cuda()
    # or
    x = x.to('cuda')
```

### 2. Autograd

PyTorch's automatic differentiation engine that powers neural network training.

```python
# Enable gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

# Compute gradients
y.backward()
print(x.grad)  # dy/dx = 2x = 4.0
```

---

## Tensors

### Creating Tensors

```python
import torch

# From Python lists
tensor1 = torch.tensor([[1, 2], [3, 4]])

# Zeros and ones
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
identity = torch.eye(3)

# Random tensors
random_uniform = torch.rand(2, 3)  # Uniform [0, 1)
random_normal = torch.randn(2, 3)  # Normal distribution
random_int = torch.randint(0, 10, (3, 4))  # Random integers

# From NumPy
import numpy as np
np_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(np_array)

# Like another tensor
x = torch.randn(3, 4)
y = torch.zeros_like(x)
z = torch.ones_like(x)
```

### Tensor Operations

```python
# Arithmetic operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

c = a + b  # Element-wise addition
d = a * b  # Element-wise multiplication
e = a @ b.T  # Dot product (for vectors)

# Matrix operations
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.matmul(A, B)  # Matrix multiplication
# or
C = A @ B

# Reshaping
x = torch.randn(12)
y = x.view(3, 4)  # Reshape to 3x4
z = x.reshape(2, 6)  # Alternative

# Indexing and slicing
tensor = torch.randn(4, 5)
row = tensor[0, :]  # First row
col = tensor[:, 1]  # Second column
subset = tensor[1:3, 2:4]  # Slicing
```

### GPU Operations

```python
# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")

# Move tensors to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(1000, 1000).to(device)

# Create directly on GPU
y = torch.randn(1000, 1000, device=device)

# Move back to CPU
x_cpu = x.cpu()
```

---

## Autograd

PyTorch's autograd system automatically computes gradients for backpropagation.

### Basic Usage

```python
import torch

# Enable gradient tracking
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward pass
z = x**2 + y**3
print(f"z = {z}")

# Backward pass
z.backward()

# Gradients
print(f"dz/dx = {x.grad}")  # 2x = 4.0
print(f"dz/dy = {y.grad}")  # 3y^2 = 27.0
```

### Gradient Context

```python
# Disable gradient tracking for inference
with torch.no_grad():
    y = x * 2
    # No gradients computed

# Or use eval mode for models
model.eval()

# Temporarily enable gradients
with torch.enable_grad():
    # Gradients enabled here
    pass
```

### Detaching Tensors

```python
# Detach from computation graph
x = torch.randn(3, requires_grad=True)
y = x.detach()  # y has no gradient
z = x.detach().requires_grad_()  # Create new leaf with gradients
```

---

## Building Neural Networks

### Using nn.Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model
model = SimpleNN(input_size=784, hidden_size=128, num_classes=10)
print(model)
```

### Convolutional Neural Network

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv layers with pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

model = CNN(num_classes=10)
```

### Sequential API

```python
# Simple sequential model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10)
)
```

---

## Training Loop

### Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Create dummy data
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))

# Create dataset and dataloader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model, loss, optimizer
model = SimpleNN(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()  # Set to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
```

### Validation Loop

```python
def validate(model, val_loader, criterion, device):
    model.eval()  # Set to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy
```

---

## Advanced Features

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step decay
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Reduce on plateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# In training loop
for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)  # For ReduceLROnPlateau
    # scheduler.step()  # For others
```

### Data Augmentation

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Transfer Learning

```python
from torchvision import models

# Load pretrained model
model = models.resnet50(pretrained=True)

# Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Only train the final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

### Model Saving and Loading

```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Save full checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load model
model = SimpleNN(784, 128, 10)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

---

## Best Practices

### 1. Always Set Model Mode
```python
model.train()  # For training
model.eval()   # For evaluation/inference
```

### 2. Use torch.no_grad() for Inference
```python
with torch.no_grad():
    predictions = model(inputs)
```

### 3. Clear Gradients
```python
optimizer.zero_grad()  # Before backward pass
```

### 4. Move Data and Model to Same Device
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
inputs = inputs.to(device)
```

### 5. Use DataLoader for Batching
```python
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

### 6. Monitor GPU Memory
```python
print(torch.cuda.memory_allocated() / 1024**2, "MB")
print(torch.cuda.memory_reserved() / 1024**2, "MB")
torch.cuda.empty_cache()  # Clear cache if needed
```

---

## Common Patterns

### Custom Dataset

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = os.listdir(img_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

### Gradient Clipping

```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Resources

### Official Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Examples](https://github.com/pytorch/examples)

### Learning Resources
- [Deep Learning with PyTorch (Book)](https://pytorch.org/deep-learning-with-pytorch)
- [PyTorch for Deep Learning (Course)](https://www.udacity.com/course/deep-learning-pytorch--ud188)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### Community
- [PyTorch Forums](https://discuss.pytorch.org/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)
- [PyTorch Slack](https://pytorch.slack.com/)

---

**Next Steps**: Try the [Building Your First Neural Network](../tutorials/first-neural-network.md) tutorial!
