# AI_ML_Learning
- [Google ScholarAI](https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=eng_artificialintelligence)
- [Anthropic for Devs](https://github.com/anthropics)
- [Spec Driven Dev with markdown](https://github.blog/ai-and-ml/generative-ai/spec-driven-development-using-markdown-as-a-programming-language-when-building-with-ai/)
- [Reasoning with Sampling](https://aakaran.github.io/reasoning_with_sampling/)
- [Claude Code News](https://www.anthropic.com/news)
- [Using Haiku as a Subagent](https://github.com/anthropics/claude-cookbooks/blob/main/multimodal/using_sub_agents.ipynb)
- [Self Adapting Language Models](https://arxiv.org/abs/2506.10943)
- [NanoChat OpenAI Founders opensource](https://github.com/karpathy/nanochat)
- [Spec Driven Dev with AI](https://github.blog/ai-and-ml/generative-ai/spec-driven-development-with-ai-get-started-with-a-new-open-source-toolkit/)
- [GROUNDING WITH GEMINI](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Grounding.ipynb)
- [EBOOK LLM PARSER/READER](https://github.com/lfnovo/open-notebook)
- [MINIMIND](https://github.com/jingyaogong/minimind/tree/master)
- [QWEN CHAT](https://chat.qwen.ai/?models=qwen3-vl-32b)
- [BIX Bench](https://huggingface.co/datasets/futurehouse/BixBench)
- [Bix Bench Arxiv](https://arxiv.org/abs/2503.00096)
- [IBM Granite](https://huggingface.co/ibm-granite)
- [Kaggle Playbook NVIDIA](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [Adventures in Triton Kernels](https://ut21.github.io/blog/triton.html)
- [Hierachecal Reasoning Model ARXIV](https://arxiv.org/abs/2506.21734)
- [Weivate Fine Tuning](https://weaviate.io/blog/fine-tune-embedding-model)
- [Routine: A Structural Planning Framework for LLM Agent System in Enterprise](https://arxiv.org/abs/2507.14447)
- 
A comprehensive collection of essential repositories and documentation for AI/ML developers, plus a complete Python environment for training machine learning models.

## Table of Contents
- [Setup Guide](#setup-guide)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [📚 Documentation](#-documentation)
- [Quick Links](#quick-links)
- [Frameworks & Libraries](#frameworks--libraries)
- [Large Language Models](#large-language-models)
- [AI Agents & Tools](#ai-agents--tools)
- [Computer Vision](#computer-vision)
- [Natural Language Processing](#natural-language-processing)
- [Reinforcement Learning](#reinforcement-learning)
- [MLOps & Production](#mlops--production)
- [Educational Resources](#educational-resources)
- [Documentation & References](#documentation--references)
- [Research Papers & Implementations](#research-papers--implementations)

---

## Setup Guide

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/deastrobooking/AI_ML_Learning.git
cd AI_ML_Learning
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
# Install core dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt

# Or install in editable mode
pip install -e .
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Project Structure

```
AI_ML_Learning/
├── configs/              # Configuration files
│   └── train_config.yaml
├── data/                 # Data directory
│   ├── raw/             # Raw data
│   └── processed/       # Processed data
├── models/              # Model storage
│   ├── saved/           # Trained models
│   └── checkpoints/     # Training checkpoints
├── notebooks/           # Jupyter notebooks
├── scripts/             # Training and utility scripts
│   ├── train_image_classifier.py
│   └── train_sklearn_model.py
├── src/                 # Source code
│   ├── __init__.py
│   ├── data_utils.py    # Data loading utilities
│   ├── models.py        # Model architectures
│   └── training.py      # Training utilities
├── tests/               # Unit tests
│   └── test_models.py
├── logs/                # Training logs
├── .env.example         # Environment variables template
├── .gitignore
├── requirements.txt     # Python dependencies
├── requirements-dev.txt # Development dependencies
├── setup.py            # Package setup
└── README.md
```

---

## Quick Start

### Training an Image Classifier

```bash
# Using PyTorch with a simple CNN
python scripts/train_image_classifier.py \
    --data_dir ./data/raw \
    --model simple_cnn \
    --num_classes 10 \
    --batch_size 32 \
    --epochs 20 \
    --lr 0.001

# Using a pretrained ResNet
python scripts/train_image_classifier.py \
    --data_dir ./data/raw \
    --model resnet \
    --num_classes 10 \
    --batch_size 32 \
    --epochs 10 \
    --lr 0.0001
```

### Training on Tabular Data

```bash
python scripts/train_sklearn_model.py \
    --data_path ./data/raw/dataset.csv \
    --target_col target \
    --model random_forest \
    --experiment_name my_experiment
```

### Using the Python API

```python
import torch
from src.models import get_model
from src.training import Trainer, get_optimizer

# Create model
model = get_model('simple_cnn', num_classes=10)

# Setup training
optimizer = get_optimizer(model, 'adam', lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Train (assuming you have train_loader and val_loader)
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device='cuda'
)

history = trainer.fit(num_epochs=10)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest tests/ --cov=src
```

### Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

---

## 📚 Documentation

Comprehensive guides and tutorials for learning AI/ML are available in the `/docs` directory!

### 🎓 For Beginners
- **[Beginner's Guide to AI/ML](docs/guides/beginners-guide.md)** - Start here if you're new!
- [Neural Networks 101](docs/guides/neural-networks-101.md)
- [Setting Up Your Environment](docs/guides/environment-setup.md)

### 🔧 Framework Guides
- **[PyTorch Deep Dive](docs/frameworks/pytorch.md)** - Complete PyTorch guide
- **[LangChain & LangGraph](docs/frameworks/langchain-langraph.md)** - Build LLM applications
- **[Hugging Face Transformers](docs/frameworks/huggingface.md)** - State-of-the-art NLP
- [TensorFlow Guide](docs/frameworks/tensorflow.md)
- [Scikit-Learn](docs/frameworks/sklearn.md)
- [FastAI](docs/frameworks/fastai.md)

### 🎯 Specialized Topics
- [Large Language Models (LLMs)](docs/guides/llms.md)
- [Computer Vision](docs/guides/computer-vision.md)
- [Natural Language Processing](docs/guides/nlp.md)
- [Reinforcement Learning](docs/guides/reinforcement-learning.md)
- [MLOps & Deployment](docs/guides/mlops.md)

### 📝 Hands-On Tutorials
- [Building Your First Neural Network](docs/tutorials/first-neural-network.md)
- [Fine-Tuning LLMs](docs/tutorials/fine-tuning-llms.md)
- [Creating AI Agents](docs/tutorials/ai-agents.md)
- [Image Classification Project](docs/tutorials/image-classification.md)
- [Deploying Models to Production](docs/tutorials/model-deployment.md)

**📖 [View Full Documentation Index](docs/README.md)**

---

---

## Quick Links
- [Microsoft Agent Framework](https://github.com/microsoft/ai-agents-for-beginners/tree/main/14-microsoft-agent-framework)
- [Github Spec Kit](https://github.com/github/spec-kit)
- [Excel Copilot Completion](https://techcommunity.microsoft.com/blog/excelblog/introducing-formula-completion---a-new-way-to-write-formulas-in-excel-using-copi/4452196)
- [Langchain Security](https://blog.langchain.com/agent-authorization-explainer/)
- [Jamba 3b model](https://www.ai21.com/blog/introducing-jamba-reasoning-3b/)

---

## Frameworks & Libraries

### Deep Learning Frameworks
1. [TensorFlow](https://github.com/tensorflow/tensorflow) - Google's open-source machine learning framework
2. [PyTorch](https://github.com/pytorch/pytorch) - Facebook's deep learning framework
3. [Keras](https://github.com/keras-team/keras) - High-level neural networks API
4. [JAX](https://github.com/google/jax) - Google's composable transformations of Python+NumPy
5. [MXNet](https://github.com/apache/mxnet) - Apache's flexible deep learning framework
6. [Caffe](https://github.com/BVLC/caffe) - Deep learning framework by Berkeley AI Research
7. [Theano](https://github.com/Theano/Theano) - Python library for deep learning
8. [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - Baidu's deep learning platform

### ML Libraries
9. [Scikit-learn](https://github.com/scikit-learn/scikit-learn) - Machine learning in Python
10. [XGBoost](https://github.com/dmlc/xgboost) - Gradient boosting framework
11. [LightGBM](https://github.com/microsoft/LightGBM) - Microsoft's gradient boosting framework
12. [CatBoost](https://github.com/catboost/catboost) - Yandex's gradient boosting library
13. [FastAI](https://github.com/fastai/fastai) - Deep learning library built on PyTorch
14. [ONNX](https://github.com/onnx/onnx) - Open Neural Network Exchange

---

## Large Language Models

### LLM Frameworks & Tools
15. [LangChain](https://github.com/langchain-ai/langchain) - Building applications with LLMs
16. [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework for LLM applications
17. [Transformers](https://github.com/huggingface/transformers) - Hugging Face's NLP library
18. [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference and serving
19. [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui) - Gradio web UI for LLMs
20. [LocalAI](https://github.com/mudler/LocalAI) - OpenAI alternative for local deployment

### Model Repositories
21. [Llama](https://github.com/meta-llama/llama) - Meta's LLaMA models
22. [GPT4All](https://github.com/nomic-ai/gpt4all) - Run LLMs locally
23. [Ollama](https://github.com/ollama/ollama) - Get up and running with large language models
24. [Mistral](https://github.com/mistralai/mistral-src) - Mistral AI models
25. [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) - Stanford's instruction-following LLM
26. [Vicuna](https://github.com/lm-sys/FastChat) - Open-source chatbot trained by fine-tuning LLaMA

---

## AI Agents & Tools

### Agent Frameworks
27. [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) - Autonomous GPT-4 experiment
28. [BabyAGI](https://github.com/yoheinakajima/babyagi) - AI-powered task management
29. [LangGraph](https://github.com/langchain-ai/langgraph) - Build stateful, multi-actor applications
30. [CrewAI](https://github.com/joaomdmoura/crewAI) - Framework for orchestrating AI agents
31. [AutoGen](https://github.com/microsoft/autogen) - Microsoft's multi-agent conversation framework
32. [Semantic Kernel](https://github.com/microsoft/semantic-kernel) - Microsoft's SDK for AI orchestration
33. [Haystack](https://github.com/deepset-ai/haystack) - NLP framework for building AI applications

### AI Tools & Applications
34. [OpenAI Cookbook](https://github.com/openai/openai-cookbook) - Examples and guides for OpenAI API
35. [Anthropic Claude](https://github.com/anthropics/anthropic-sdk-python) - Claude SDK
36. [ChatGPT](https://github.com/acheong08/ChatGPT) - Reverse engineered ChatGPT API
37. [Copilot](https://github.com/features/copilot) - GitHub Copilot
38. [Continue](https://github.com/continuedev/continue) - Open-source autopilot for VS Code

---

## Computer Vision

### CV Libraries & Tools
39. [OpenCV](https://github.com/opencv/opencv) - Computer vision library
40. [YOLO](https://github.com/ultralytics/ultralytics) - Real-time object detection
41. [Detectron2](https://github.com/facebookresearch/detectron2) - Facebook's object detection platform
42. [MMDetection](https://github.com/open-mmlab/mmdetection) - OpenMMLab detection toolbox
43. [Segment Anything](https://github.com/facebookresearch/segment-anything) - Meta's segmentation model
44. [MediaPipe](https://github.com/google/mediapipe) - Google's ML solutions
45. [CLIP](https://github.com/openai/CLIP) - OpenAI's vision-language model
46. [Kornia](https://github.com/kornia/kornia) - Differentiable computer vision library

### Image Generation
47. [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) - Text-to-image generation
48. [DALL-E](https://github.com/openai/dall-e) - OpenAI's image generation
49. [Midjourney](https://github.com/midjourney/midjourney) - AI art generation
50. [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Stable Diffusion GUI
51. [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - Stable Diffusion web UI
52. [ControlNet](https://github.com/lllyasviel/ControlNet) - Controlling image generation

---

## Natural Language Processing

### NLP Tools
53. [spaCy](https://github.com/explosion/spaCy) - Industrial-strength NLP
54. [NLTK](https://github.com/nltk/nltk) - Natural Language Toolkit
55. [Gensim](https://github.com/RaRe-Technologies/gensim) - Topic modeling library
56. [AllenNLP](https://github.com/allenai/allennlp) - NLP research library
57. [Stanza](https://github.com/stanfordnlp/stanza) - Stanford NLP toolkit
58. [TextBlob](https://github.com/sloria/TextBlob) - Simple NLP library

### Text Processing
59. [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) - Sentence embeddings
60. [BERTopic](https://github.com/MaartenGr/BERTopic) - Topic modeling with BERT
61. [Flair](https://github.com/flairNLP/flair) - NLP framework
62. [Tokenizers](https://github.com/huggingface/tokenizers) - Fast tokenizers

---

## Reinforcement Learning

63. [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms in PyTorch
64. [OpenAI Gym](https://github.com/openai/gym) - RL toolkit
65. [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) - Maintained fork of OpenAI Gym
66. [RLlib](https://github.com/ray-project/ray) - Scalable RL library
67. [TF-Agents](https://github.com/tensorflow/agents) - TensorFlow RL library
68. [Tianshou](https://github.com/thu-ml/tianshou) - PyTorch RL framework
69. [CleanRL](https://github.com/vwxyzjn/cleanrl) - High-quality single-file RL implementations

---

## MLOps & Production

### ML Engineering
70. [MLflow](https://github.com/mlflow/mlflow) - ML lifecycle platform
71. [Kubeflow](https://github.com/kubeflow/kubeflow) - ML toolkit for Kubernetes
72. [DVC](https://github.com/iterative/dvc) - Data version control
73. [Weights & Biases](https://github.com/wandb/wandb) - ML experiment tracking
74. [TensorBoard](https://github.com/tensorflow/tensorboard) - TensorFlow visualization toolkit
75. [Metaflow](https://github.com/Netflix/metaflow) - Netflix's ML infrastructure

### Model Serving
76. [TorchServe](https://github.com/pytorch/serve) - PyTorch model serving
77. [TensorFlow Serving](https://github.com/tensorflow/serving) - TensorFlow model serving
78. [BentoML](https://github.com/bentoml/BentoML) - ML model serving framework
79. [Seldon Core](https://github.com/SeldonIO/seldon-core) - ML deployment on Kubernetes
80. [KServe](https://github.com/kserve/kserve) - Serverless inference on Kubernetes
81. [Ray Serve](https://github.com/ray-project/ray) - Scalable model serving

---

## Educational Resources

### Learning & Tutorials
82. [ML For Beginners](https://github.com/microsoft/ML-For-Beginners) - Microsoft's ML curriculum
83. [Deep Learning Book](https://github.com/janishar/mit-deep-learning-book-pdf) - Ian Goodfellow's book
84. [Dive into Deep Learning](https://github.com/d2l-ai/d2l-en) - Interactive deep learning book
85. [Fast.ai Course](https://github.com/fastai/course-v4) - Practical deep learning
86. [TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples) - TensorFlow tutorials
87. [PyTorch Examples](https://github.com/pytorch/examples) - PyTorch tutorials
88. [Machine Learning Yearning](https://github.com/ajaymache/machine-learning-yearning) - Andrew Ng's book
89. [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning) - Curated ML resources

---

## Documentation & References

### Official Documentation
90. [TensorFlow Docs](https://www.tensorflow.org/api_docs) - TensorFlow API documentation
91. [PyTorch Docs](https://pytorch.org/docs/stable/index.html) - PyTorch documentation
92. [Hugging Face Docs](https://huggingface.co/docs) - Hugging Face documentation
93. [OpenAI Docs](https://platform.openai.com/docs) - OpenAI API documentation
94. [Anthropic Docs](https://docs.anthropic.com/) - Claude API documentation
95. [Google AI](https://ai.google/tools/) - Google AI tools and documentation
96. [Papers with Code](https://paperswithcode.com/) - ML papers with implementation
97. [Arxiv Sanity](https://arxiv-sanity-lite.com/) - Browse research papers

### Community & Resources
98. [Kaggle](https://www.kaggle.com/) - Data science competitions and datasets
99. [Towards Data Science](https://towardsdatascience.com/) - ML/AI blog platform
100. [Distill.pub](https://distill.pub/) - Research journal for ML
101. [AI Alignment Forum](https://www.alignmentforum.org/) - AI safety research
102. [MLOps Community](https://mlops.community/) - MLOps resources and community

---

## Research Papers & Implementations

### Important Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [BERT](https://arxiv.org/abs/1810.04805) - Bidirectional transformers
- [GPT-3](https://arxiv.org/abs/2005.14165) - Language models
- [ResNet](https://arxiv.org/abs/1512.03385) - Deep residual learning
- [GAN](https://arxiv.org/abs/1406.2661) - Generative adversarial networks

### Implementation Repositories
- [Papers We Love](https://github.com/papers-we-love/papers-we-love) - Academic CS papers
- [Annotated Transformer](https://github.com/harvardnlp/annotated-transformer) - Line-by-line implementation
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) - Diffusion models explained

---

## Contributing

Feel free to submit pull requests to add more resources or update existing links!

## License

This is a curated list for educational purposes. Please check individual repositories for their respective licenses.
