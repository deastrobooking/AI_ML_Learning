# Hugging Face Transformers Guide

Master the Hugging Face ecosystem for state-of-the-art NLP and beyond.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Pipeline API](#pipeline-api)
4. [Loading Models](#loading-models)
5. [Fine-Tuning](#fine-tuning)
6. [Custom Models](#custom-models)
7. [Datasets](#datasets)
8. [Inference Optimization](#inference-optimization)
9. [Common Use Cases](#common-use-cases)
10. [Resources](#resources)

---

## Introduction

Hugging Face Transformers provides thousands of pretrained models for:
- **Natural Language Processing**: text classification, NER, QA, summarization
- **Computer Vision**: image classification, object detection, segmentation
- **Audio**: speech recognition, audio classification
- **Multimodal**: vision-language models, image captioning

### Key Features
- 🚀 Easy-to-use APIs
- 🤗 Huge model hub with 100,000+ models
- 🔥 PyTorch, TensorFlow, and JAX support
- ⚡ Optimized inference
- 🎯 Fine-tuning made simple

---

## Installation & Setup

```bash
# Basic installation
pip install transformers

# With PyTorch
pip install transformers torch

# With TensorFlow
pip install transformers tensorflow

# Additional tools
pip install datasets  # For loading datasets
pip install accelerate  # For distributed training
pip install bitsandbytes  # For quantization
pip install sentencepiece  # For some tokenizers
```

### Verify Installation

```python
import transformers
print(f"Transformers version: {transformers.__version__}")

# Check available models
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
print(classifier("I love Hugging Face!"))
```

---

## Pipeline API

The simplest way to use models - just specify the task!

### Text Classification

```python
from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("This product is amazing!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Multi-class classification
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
results = classifier([
    "I love this!",
    "I hate this.",
    "This is okay."
])
```

### Named Entity Recognition

```python
ner = pipeline("ner", grouped_entities=True)
result = ner("Hugging Face is based in New York City.")
print(result)
# [{'entity_group': 'ORG', 'word': 'Hugging Face', ...},
#  {'entity_group': 'LOC', 'word': 'New York City', ...}]
```

### Question Answering

```python
qa = pipeline("question-answering")
context = "Hugging Face is a company that develops tools for building applications using machine learning."
question = "What does Hugging Face do?"

result = qa(question=question, context=context)
print(result['answer'])
```

### Text Generation

```python
generator = pipeline("text-generation", model="gpt2")
result = generator(
    "Once upon a time",
    max_length=50,
    num_return_sequences=2,
    temperature=0.7
)
print(result[0]['generated_text'])
```

### Translation

```python
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
print(result[0]['translation_text'])
# "Bonjour, comment allez-vous?"
```

### Summarization

```python
summarizer = pipeline("summarization")
article = """
The Hugging Face Transformers library provides thousands of pretrained models
to perform tasks on texts such as classification, information extraction,
question answering, summarization, translation, text generation, etc.
"""
summary = summarizer(article, max_length=50, min_length=10)
print(summary[0]['summary_text'])
```

### Zero-Shot Classification

```python
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about Python programming.",
    candidate_labels=["education", "politics", "business"]
)
print(result)
```

---

## Loading Models

### Basic Model Loading

```python
from transformers import AutoTokenizer, AutoModel

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenize input
text = "Hello, world!"
inputs = tokenizer(text, return_tensors="pt")

# Forward pass
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

### Task-Specific Models

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)

# For classification
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

# For text generation
model = AutoModelForCausalLM.from_pretrained("gpt2")

# For sequence-to-sequence (translation, summarization)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
```

### Advanced Loading Options

```python
from transformers import AutoModel
import torch

# Load on specific device
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    device_map="auto",  # Automatic device placement
    torch_dtype=torch.float16  # Use half precision
)

# Load with quantization
model = AutoModel.from_pretrained(
    "facebook/opt-6.7b",
    load_in_8bit=True,  # 8-bit quantization
    device_map="auto"
)

# Load specific checkpoint
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    revision="main",  # Git branch/tag
    cache_dir="./model_cache"
)
```

---

## Fine-Tuning

### Basic Fine-Tuning Script

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

# Load dataset
dataset = load_dataset("imdb")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# Train!
trainer.train()

# Evaluate
results = trainer.evaluate()
print(results)

# Save model
trainer.save_model("./my-fine-tuned-model")
```

### Parameter-Efficient Fine-Tuning (PEFT)

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)

# Apply LoRA to model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 109,483,778 || trainable%: 0.27%

# Train as usual
trainer = Trainer(model=model, ...)
trainer.train()
```

---

## Custom Models

### Creating a Custom Model

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class CustomBertClassifier(nn.Module):
    def __init__(self, num_labels, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

# Use the custom model
model = CustomBertClassifier(num_labels=3)
```

### Custom Tokenizer

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Create tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# Train tokenizer
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["path/to/corpus.txt"], trainer=trainer)

# Save tokenizer
tokenizer.save("custom_tokenizer.json")

# Load with Transformers
from transformers import PreTrainedTokenizerFast
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="custom_tokenizer.json")
```

---

## Datasets

### Loading Datasets

```python
from datasets import load_dataset

# Load from Hugging Face Hub
dataset = load_dataset("imdb")
dataset = load_dataset("squad")
dataset = load_dataset("glue", "mrpc")

# Load local files
dataset = load_dataset("csv", data_files="data.csv")
dataset = load_dataset("json", data_files="data.json")

# Load from pandas
import pandas as pd
from datasets import Dataset

df = pd.read_csv("data.csv")
dataset = Dataset.from_pandas(df)
```

### Dataset Operations

```python
# View dataset
print(dataset)
print(dataset["train"][0])

# Map function to dataset
def preprocess(example):
    example["text"] = example["text"].lower()
    return example

processed = dataset.map(preprocess)

# Filter dataset
filtered = dataset.filter(lambda x: len(x["text"]) > 100)

# Split dataset
train_test = dataset.train_test_split(test_size=0.2)

# Batch processing
def tokenize_batch(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenized = dataset.map(tokenize_batch, batched=True)
```

---

## Inference Optimization

### Batched Inference

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", device=0)  # GPU

# Batch processing
texts = ["Text 1", "Text 2", "Text 3", ...]
results = classifier(texts, batch_size=32)
```

### Optimum & ONNX

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Convert to ONNX
model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    export=True
)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Faster inference
inputs = tokenizer("This is great!", return_tensors="pt")
outputs = model(**inputs)
```

### Dynamic Quantization

```python
import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Quantize model
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Smaller size, faster inference
torch.save(quantized_model.state_dict(), "quantized_model.pt")
```

---

## Common Use Cases

### Embedding Extraction

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

embeddings = get_embeddings(["Hello world", "Hugging Face"])
```

### Multi-GPU Inference

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "gpt2-large",
    device_map="auto",  # Automatically distribute across GPUs
    torch_dtype=torch.float16
)
```

### Streaming Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

streamer = TextStreamer(tokenizer)

inputs = tokenizer("Once upon a time", return_tensors="pt")
model.generate(**inputs, streamer=streamer, max_length=100)
```

---

## Resources

### Official Documentation
- [Transformers Docs](https://huggingface.co/docs/transformers)
- [Model Hub](https://huggingface.co/models)
- [Datasets Hub](https://huggingface.co/datasets)
- [Course](https://huggingface.co/course)

### Learning Resources
- [NLP Course](https://huggingface.co/learn/nlp-course)
- [Fine-tuning Tutorial](https://huggingface.co/docs/transformers/training)
- [Example Scripts](https://github.com/huggingface/transformers/tree/main/examples)

### Community
- [Forum](https://discuss.huggingface.co/)
- [Discord](https://discord.gg/hugging-face)
- [Twitter](https://twitter.com/huggingface)

---

**Next Steps**: Try [Fine-Tuning LLMs](../tutorials/fine-tuning-llms.md) tutorial!
