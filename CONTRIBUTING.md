# Contributing to AI_ML_Learning

Thank you for your interest in contributing to AI_ML_Learning! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/AI_ML_Learning.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Set up the development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

## Development Workflow

### Code Style

We follow PEP 8 style guidelines. Use the following tools to maintain code quality:

```bash
# Format code
make format

# Run linters
make lint
```

### Testing

All new features should include tests. Run tests before submitting a PR:

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_models.py -v
```

### Adding New Models

1. Add your model class to `src/models.py`
2. Include docstrings explaining the model architecture
3. Add a factory method in `get_model()` function
4. Write unit tests in `tests/test_models.py`
5. Add usage examples in docstrings

Example:
```python
class YourNewModel(nn.Module):
    """
    Brief description of the model.
    
    Args:
        param1: Description
        param2: Description
    """
    def __init__(self, param1, param2):
        super().__init__()
        # Your implementation
        
    def forward(self, x):
        # Your implementation
        return x
```

### Adding Training Scripts

1. Create a new script in `scripts/` directory
2. Use argparse for command-line arguments
3. Follow the structure of existing scripts
4. Document all parameters
5. Include example usage in comments

### Adding Notebooks

1. Create notebooks in the `notebooks/` directory
2. Use descriptive names (e.g., `02_transfer_learning.ipynb`)
3. Clear all outputs before committing
4. Include markdown cells explaining each step
5. Keep notebooks focused on a single topic

## Commit Guidelines

### Commit Messages

Follow the conventional commits format:

- `feat: add new CNN architecture`
- `fix: correct data loading bug`
- `docs: update README with setup instructions`
- `test: add tests for data_utils`
- `refactor: improve training loop efficiency`
- `style: format code with black`

### Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation if you're changing functionality
3. Add tests for new features
4. Ensure all tests pass
5. Update the version number if appropriate
6. Create a Pull Request with a clear description

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe the tests you ran

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

## Adding Resources to README

When adding new repositories or documentation links:

1. Ensure the resource is actively maintained
2. Add it to the appropriate category
3. Include a brief description
4. Verify the link is correct
5. Check for duplicates

## Questions or Issues?

- Open an issue for bugs or feature requests
- Use discussions for questions and general chat
- Tag your issues appropriately

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to AI_ML_Learning! 🚀
