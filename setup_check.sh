#!/bin/bash

# AI/ML Learning Environment Setup Script
# This script helps you set up the environment and verify the installation

echo "=================================="
echo "AI/ML Learning Setup Verification"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1)
echo "✓ $python_version"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment is activated: $VIRTUAL_ENV"
else
    echo "⚠ Warning: No virtual environment detected"
    echo "  Recommendation: Create and activate a virtual environment"
    echo "  Run: python -m venv venv && source venv/bin/activate"
fi
echo ""

# Check if requirements are installed
echo "Checking installed packages..."
packages=("torch" "tensorflow" "numpy" "pandas" "scikit-learn" "transformers")

for package in "${packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        version=$(python -c "import $package; print($package.__version__)" 2>/dev/null)
        echo "✓ $package ($version)"
    else
        echo "✗ $package (not installed)"
    fi
done
echo ""

# Check CUDA availability
echo "Checking GPU/CUDA availability..."
cuda_available=$(python -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')" 2>/dev/null)
if [ "$cuda_available" = "Yes" ]; then
    cuda_version=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null)
    echo "✓ CUDA is available"
    echo "  CUDA Version: $cuda_version"
    echo "  GPU Count: $gpu_count"
    echo "  GPU Name: $gpu_name"
else
    echo "⚠ CUDA is not available (CPU mode)"
    echo "  Training will be slower without GPU"
fi
echo ""

# Check directory structure
echo "Checking directory structure..."
dirs=("data/raw" "data/processed" "models/saved" "models/checkpoints" "logs" "notebooks" "scripts" "src" "tests")

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir"
    else
        echo "✗ $dir (missing)"
    fi
done
echo ""

# Run simple test
echo "Running simple functionality test..."
python -c "
import sys
sys.path.append('.')
from src.models import SimpleCNN
import torch

model = SimpleCNN(num_classes=10)
x = torch.randn(1, 3, 224, 224)
output = model(x)
print('✓ Model creation and forward pass successful')
print(f'  Output shape: {output.shape}')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
else
    echo "✗ Model test failed"
    echo "  Make sure all dependencies are installed"
fi

# Summary
echo "=================================="
echo "Setup Verification Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and configure your API keys"
echo "2. Explore the notebooks/ directory for examples"
echo "3. Try running: python scripts/train_image_classifier.py --help"
echo "4. Read CONTRIBUTING.md if you want to contribute"
echo ""
echo "For help, visit: https://github.com/deastrobooking/AI_ML_Learning"
