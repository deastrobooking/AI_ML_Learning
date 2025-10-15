"""
PyTorch model architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SimpleCNN(nn.Module):
    """Simple CNN for image classification"""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


class ResNetClassifier(nn.Module):
    """ResNet-based classifier with pretrained weights"""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super(ResNetClassifier, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Replace final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)


class SimpleMLPClassifier(nn.Module):
    """Simple Multi-Layer Perceptron for tabular data"""
    
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int, dropout: float = 0.3):
        super(SimpleMLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class TransformerClassifier(nn.Module):
    """Simple Transformer-based classifier"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, num_classes: int = 2, max_seq_length: int = 512):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Embedding(max_seq_length, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        
        x = self.embedding(x) + self.pos_encoder(positions)
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, embed_dim)
        
        x = self.transformer(x)
        x = x.mean(dim=0)  # Global average pooling
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def get_model(model_name: str, **kwargs):
    """
    Factory function to get model by name
    
    Args:
        model_name: Name of the model
        **kwargs: Model-specific arguments
        
    Returns:
        PyTorch model
    """
    models_dict = {
        'simple_cnn': SimpleCNN,
        'resnet': ResNetClassifier,
        'mlp': SimpleMLPClassifier,
        'transformer': TransformerClassifier,
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not found. Available: {list(models_dict.keys())}")
    
    return models_dict[model_name](**kwargs)
