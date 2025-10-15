"""
Unit tests for models
"""

import pytest
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import SimpleCNN, ResNetClassifier, SimpleMLPClassifier, TransformerClassifier


def test_simple_cnn():
    """Test SimpleCNN forward pass"""
    model = SimpleCNN(num_classes=10, input_channels=3)
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"


def test_resnet_classifier():
    """Test ResNetClassifier forward pass"""
    model = ResNetClassifier(num_classes=10, pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"


def test_mlp_classifier():
    """Test SimpleMLPClassifier forward pass"""
    model = SimpleMLPClassifier(input_dim=100, hidden_dims=[64, 32], num_classes=5)
    x = torch.randn(4, 100)
    output = model(x)
    assert output.shape == (4, 5), f"Expected shape (4, 5), got {output.shape}"


def test_transformer_classifier():
    """Test TransformerClassifier forward pass"""
    model = TransformerClassifier(vocab_size=1000, embed_dim=128, num_heads=4, num_layers=2, num_classes=3)
    x = torch.randint(0, 1000, (4, 50))
    output = model(x)
    assert output.shape == (4, 3), f"Expected shape (4, 3), got {output.shape}"


if __name__ == '__main__':
    pytest.main([__file__])
