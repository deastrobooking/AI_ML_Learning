# TensorFlow Guide

A comprehensive guide to TensorFlow 2.x, Google's open-source machine learning framework.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Tensors and Operations](#tensors-and-operations)
4. [Building Models](#building-models)
5. [Training](#training)
6. [Keras API](#keras-api)
7. [Advanced Features](#advanced-features)
8. [Production Deployment](#production-deployment)
9. [Resources](#resources)

---

## Introduction

TensorFlow is an end-to-end platform for machine learning with:
- **Eager Execution**: Intuitive and Pythonic (TF 2.x default)
- **Keras Integration**: High-level API for fast prototyping
- **Production Ready**: TensorFlow Serving, TF Lite, TF.js
- **Distributed Training**: Easy multi-GPU and TPU support
- **TensorBoard**: Powerful visualization tool

---

## Installation

```bash
# CPU version
pip install tensorflow

# GPU version (CUDA required)
pip install tensorflow[and-cuda]

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

---

## Tensors and Operations

### Creating Tensors

```python
import tensorflow as tf
import numpy as np

# From Python lists
tensor1 = tf.constant([[1, 2], [3, 4]])
print(tensor1)

# Different dtypes
float_tensor = tf.constant([1.0, 2.0], dtype=tf.float32)
int_tensor = tf.constant([1, 2], dtype=tf.int32)

# Zeros and ones
zeros = tf.zeros([3, 4])
ones = tf.ones([2, 3])

# Random tensors
random_normal = tf.random.normal([3, 3], mean=0.0, stddev=1.0)
random_uniform = tf.random.uniform([2, 2], minval=0, maxval=10)

# From NumPy
np_array = np.array([1, 2, 3])
tensor_from_numpy = tf.constant(np_array)
```

### Tensor Operations

```python
# Arithmetic
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

c = tf.add(a, b)  # or a + b
d = tf.multiply(a, b)  # or a * b

# Matrix operations
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])

C = tf.matmul(A, B)  # Matrix multiplication

# Reshaping
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
reshaped = tf.reshape(tensor, [3, 2])

# Slicing
tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row = tensor[0, :]  # First row
col = tensor[:, 1]  # Second column
subset = tensor[1:, :2]  # Slicing
```

### Gradients

```python
# Automatic differentiation
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

dy_dx = tape.gradient(y, x)
print(f"dy/dx at x=3: {dy_dx}")  # 6.0
```

---

## Building Models

### Sequential API (Simple)

```python
from tensorflow import keras
from tensorflow.keras import layers

# Simple sequential model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.summary()
```

### Functional API (Flexible)

```python
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

### Subclassing (Most Flexible)

```python
class CustomModel(keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(64, activation='relu')
        self.dropout2 = layers.Dropout(0.2)
        self.dense3 = layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = self.dropout1(x)
        x = self.dense2(x)
        if training:
            x = self.dropout2(x)
        return self.dense3(x)

model = CustomModel()
```

### CNN Example

```python
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

---

## Training

### Complete Training Example

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Build model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Predict
predictions = model.predict(x_test[:5])
print(np.argmax(predictions, axis=1))
```

### Custom Training Loop

```python
# Define model, loss, optimizer
model = CustomModel()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Metrics
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# Training step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_acc_metric.update_state(y, predictions)
    return loss

# Validation step
@tf.function
def val_step(x, y):
    predictions = model(x, training=False)
    val_acc_metric.update_state(y, predictions)

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Training
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, y_batch)
        
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss:.4f}")
    
    train_acc = train_acc_metric.result()
    print(f"Training accuracy: {train_acc:.4f}")
    train_acc_metric.reset_states()
    
    # Validation
    for x_batch, y_batch in val_dataset:
        val_step(x_batch, y_batch)
    
    val_acc = val_acc_metric.result()
    print(f"Validation accuracy: {val_acc:.4f}")
    val_acc_metric.reset_states()
```

---

## Keras API

### Callbacks

```python
# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Model checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    'model_{epoch:02d}.h5',
    save_best_only=True,
    monitor='val_accuracy'
)

# Learning rate scheduling
lr_schedule = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3
)

# TensorBoard
tensorboard = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1
)

# Custom callback
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch}: Loss = {logs['loss']:.4f}")

# Use callbacks
model.fit(
    x_train, y_train,
    epochs=50,
    callbacks=[early_stop, checkpoint, lr_schedule, tensorboard]
)
```

### Data Pipeline with tf.data

```python
# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Shuffle and batch
dataset = dataset.shuffle(buffer_size=1024).batch(32)

# Prefetch for performance
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Data augmentation
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    return image, label

dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

# Use in training
model.fit(dataset, epochs=10)
```

---

## Advanced Features

### Transfer Learning

```python
# Load pretrained model
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# Add custom layers
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile and train
model.compile(
    optimizer=keras.optimizers.Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune later
base_model.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Mixed Precision Training

```python
from tensorflow.keras import mixed_precision

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Build model (automatically uses mixed precision)
model = keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

# Compile with loss scaling
optimizer = keras.optimizers.Adam()
optimizer = mixed_precision.LossScaleOptimizer(optimizer)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
```

### Distributed Training

```python
# Multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Create model within strategy scope
with strategy.scope():
    model = keras.Sequential([
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Train as usual
model.fit(train_dataset, epochs=10)
```

---

## Production Deployment

### SavedModel Format

```python
# Save model
model.save('my_model')

# Load model
loaded_model = keras.models.load_model('my_model')

# Make predictions
predictions = loaded_model.predict(x_test)
```

### TensorFlow Serving

```bash
# Save model for serving
model.save('my_model/1')

# Run TensorFlow Serving
docker run -p 8501:8501 \
  --mount type=bind,source=$(pwd)/my_model,target=/models/my_model \
  -e MODEL_NAME=my_model \
  tensorflow/serving
```

```python
# Client code
import requests
import json

data = json.dumps({
    "signature_name": "serving_default",
    "instances": x_test[:5].tolist()
})

response = requests.post('http://localhost:8501/v1/models/my_model:predict', data=data)
predictions = response.json()['predictions']
```

### TensorFlow Lite (Mobile)

```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Use in mobile app
# See TFLite documentation for Android/iOS integration
```

---

## Resources

### Official Documentation
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Keras API](https://keras.io/api/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

### Learning Resources
- [TensorFlow Certificate](https://www.tensorflow.org/certificate)
- [Coursera TensorFlow Specialization](https://www.coursera.org/professional-certificates/tensorflow-in-practice)

### Community
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow)
- [GitHub](https://github.com/tensorflow/tensorflow)

---

**Next**: [Building Your First Neural Network](../tutorials/first-neural-network.md)
