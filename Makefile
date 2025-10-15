.PHONY: help install install-dev clean test train-image train-sklearn notebook lint format

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make clean         - Clean generated files"
	@echo "  make test          - Run tests"
	@echo "  make train-image   - Train image classifier (example)"
	@echo "  make train-sklearn - Train sklearn model (example)"
	@echo "  make notebook      - Start Jupyter Lab"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete
	find . -type d -name '.ipynb_checkpoints' -delete
	rm -rf build/ dist/ *.egg-info

test:
	pytest tests/ -v --cov=src --cov-report=html

train-image:
	python scripts/train_image_classifier.py \
		--data_dir ./data/raw \
		--model simple_cnn \
		--num_classes 10 \
		--batch_size 32 \
		--epochs 10 \
		--lr 0.001

train-sklearn:
	@echo "Usage: make train-sklearn DATA=path/to/data.csv TARGET=target_column"
	@echo "Example: make train-sklearn DATA=./data/raw/iris.csv TARGET=species"

notebook:
	jupyter lab

lint:
	flake8 src/ scripts/ tests/
	pylint src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/
