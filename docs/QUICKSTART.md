# Quick Start Guide - Training Your First Model

Get your first AI model running in 5 minutes!

## Choose Your Path

### Path A: Image Classification (Beginner Friendly)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open Jupyter notebook
jupyter lab

# 3. Open notebooks/01_getting_started.ipynb
# Follow along - it includes MNIST digit recognition!
```

### Path B: Using Training Scripts
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download a sample dataset or use MNIST
python -c "
from torchvision import datasets
datasets.MNIST('./data/raw', download=True)
print('Dataset downloaded!')
"

# 3. Train a model
python scripts/train_image_classifier.py \
    --model simple_cnn \
    --epochs 5 \
    --batch_size 32
```

### Path C: Interactive Python
```python
# Run this in Python or Jupyter
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data/raw', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Simple model
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

# Train
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(3):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed!")

print("Model trained! 🎉")
```

## What to Learn Next

1. **Understand the Basics**: Read [Beginner's Guide](guides/beginners-guide.md)
2. **Master PyTorch**: Study [PyTorch Deep Dive](frameworks/pytorch.md)
3. **Build with LLMs**: Explore [LangChain & LangGraph](frameworks/langchain-langraph.md)
4. **Deploy Models**: Learn [MLOps](guides/mlops.md)

## Need Help?

- 📖 Full documentation: [docs/README.md](README.md)
- 🐛 Issues: Open a GitHub issue
- 💬 Questions: Check discussions

**Happy Learning! 🚀**
