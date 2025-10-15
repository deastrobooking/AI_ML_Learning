from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai_ml_learning",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI/ML Learning Repository with Training Examples",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deastrobooking/AI_ML_Learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "tensorflow>=2.15.0",
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "transformers>=4.35.0",
        "pytorch-lightning>=2.1.0",
    ],
)
