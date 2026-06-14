# vgg-cifar10-linear-quantization
Implemented post-training linear quantization for a VGG-style CNN on CIFAR-10 using PyTorch. Quantized weights and activations to INT8, developed scale and zero-point calibration, and evaluated accuracy trade-offs between FP32 and quantized inference.

# Linear Quantization of VGG on CIFAR-10

## Overview

This project explores **Linear Quantization** techniques applied to a VGG-style Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset using PyTorch. The primary objective is to reduce model storage requirements and improve deployment efficiency by converting floating-point (FP32) weights into lower-precision integer representations while preserving classification performance.

## Key Features

* VGG-based CNN architecture for CIFAR-10 image classification
* Data augmentation using random cropping and horizontal flipping
* Baseline FP32 model evaluation
* Uniform linear quantization of network weights
* INT8 weight representation for memory-efficient storage
* Performance comparison between FP32 and quantized models
* Analysis of model compression and accuracy trade-offs

## Dataset

The model is trained and evaluated on the **CIFAR-10** dataset, a widely used benchmark for image classification tasks.

### CIFAR-10 Classes

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

## Model Architecture

The implementation utilizes a compact VGG-style CNN composed of the following components:

* Convolutional Layers
* Batch Normalization
* ReLU Activation Functions
* Max Pooling Layers
* Global Average Pooling
* Fully Connected Classification Layer

### Input

```text
3 × 32 × 32
```

### Output

```text
10 Classes
```

## Linear Quantization

Linear quantization maps floating-point weights to a finite integer range, enabling more efficient storage and computation.

### INT8 Quantization Process

```text
FP32 Weights
      ↓
Scale Computation
      ↓
INT8 Representation (-128 to 127)
```

### Quantization Formula

```text
scale = (max_weight - min_weight) / (qmax - qmin)

quantized_weight =
round(weight / scale)
```

### Dequantization Formula

```text
dequantized_weight =
quantized_weight × scale
```

During inference, quantized weights can be reconstructed using the dequantization process while benefiting from reduced memory requirements.

## Quantization Workflow

```text
CIFAR-10 Dataset
        ↓
Data Preprocessing
        ↓
VGG Model
        ↓
FP32 Evaluation
        ↓
Linear Quantization
        ↓
INT8 Model
        ↓
Performance Evaluation
```
## Experimental Results

### FP32 Baseline

The original VGG model was evaluated using floating-point (FP32) weights and activations.

| Metric     | Value      |
| ---------- | ---------- |
| Accuracy   | 92.95%     |
| Model Size | 281.63 MiB |

### INT8 Linear Quantization

Post-training linear quantization was applied to convert the network parameters from FP32 to INT8 representation.

| Metric               | Value     |
| -------------------- | --------- |
| Accuracy             | 92.78%    |
| Estimated Model Size | ~70.4 MiB |

### Comparison

| Model          | Accuracy | Model Size |
| -------------- | -------- | ---------- |
| FP32           | 92.95%   | 281.63 MiB |
| INT8 Quantized | 92.78%   | ~70.4 MiB  |

### Accuracy and Compression Analysis

* Accuracy degradation after quantization: **0.17%**
* Approximate model compression ratio: **4×**
* Significant reduction in storage requirements with negligible loss in classification performance.
* Demonstrates the effectiveness of post-training linear quantization for deploying CNNs on memory-constrained devices.

### Key Takeaways

* Linear quantization successfully reduced model size while preserving predictive performance.
* INT8 inference achieved nearly identical accuracy compared to the FP32 baseline.
* The quantized model is more suitable for edge devices and embedded AI applications where memory efficiency is critical.
* Results highlight the practical trade-off between numerical precision and deployment efficiency.

## Results

The quantized model maintains classification accuracy close to the original FP32 model while significantly reducing storage requirements.

### Observations

* Reduced model size
* Lower memory consumption
* Faster data transfer and deployment
* Minimal accuracy degradation
* Improved suitability for resource-constrained environments

## Technologies Used

* Python
* PyTorch
* TorchVision
* NumPy
* Matplotlib

## Project Structure

```text
.
├── Quantization.ipynb
└── README.md
```

## Learning Outcomes

This project provided practical experience in:

* Neural network quantization fundamentals
* FP32 and INT8 numerical representations
* Scale factor computation and weight mapping
* Post-training quantization techniques
* Accuracy versus compression trade-offs
* Efficient deployment of deep learning models

## Future Enhancements

* Per-channel quantization
* Mixed-precision quantization
* Quantization-Aware Training (QAT)
* Dynamic quantization techniques
* Weight clustering and k-means quantization

## References

* PyTorch Quantization Documentation
* CIFAR-10 Dataset Documentation
* VGG Network Architecture
* Deep Learning for Computer Vision Literature
