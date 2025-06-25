# Custom Hardware Integration with OFA

This document describes how to integrate custom hardware (NPU) with the Once-for-All (OFA) framework for quantized model deployment using ONNX Runtime.

## Overview

The custom hardware integration allows you to:
1. **Create latency predictors** for your custom NPU using ONNX Runtime
2. **Build accuracy predictors** for quantized models on your hardware
3. **Search for optimal sub-networks** tailored to your hardware constraints
4. **Evaluate quantized models** with realistic performance metrics

## Components

### 1. Quantized Latency Predictor (`QuantizedLatencyPredictor`)
- Measures actual latency of quantized ONNX models on your NPU
- Builds lookup tables for different layer configurations
- Supports dynamic input shapes and quantization schemes

### 2. Quantized Accuracy Predictor (`QuantizedAccuracyPredictor`)
- Predicts accuracy of quantized sub-networks without full evaluation
- Trained on a dataset of quantized model configurations and their accuracies
- Accounts for quantization-specific accuracy degradation

### 3. Custom Hardware Evolution Finder (`CustomHardwareEvolutionFinder`)
- Evolutionary search optimized for your hardware constraints
- Integrates both latency and accuracy predictors
- Supports multi-objective optimization (accuracy vs latency vs power)

### 4. ONNX Model Generator (`OFAToONNXConverter`)
- Converts OFA sub-networks to ONNX format
- Applies quantization schemes (INT8, FP16, etc.)
- Optimizes for your specific NPU architecture

## Quick Start

### 1. Setup

```bash
# Install additional dependencies
pip install onnx onnxruntime onnxruntime-extensions
pip install onnxoptimizer onnxsim

# For NPU-specific runtime (replace with your NPU's runtime)
# pip install your-npu-runtime
```

### 2. Basic Usage

```python
from ofa.custom_hardware import (
    QuantizedLatencyPredictor, 
    QuantizedAccuracyPredictor,
    CustomHardwareEvolutionFinder,
    OFAToONNXConverter
)

# Load OFA network
from ofa.model_zoo import ofa_net
ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)

# Setup hardware-specific predictors
latency_predictor = QuantizedLatencyPredictor(
    hardware_name='your_npu',
    quantization_scheme='int8',
    onnx_runtime_providers=['YourNPUExecutionProvider']
)

accuracy_predictor = QuantizedAccuracyPredictor(
    quantization_scheme='int8',
    pretrained=True
)

# Build latency lookup table
latency_predictor.build_lookup_table(ofa_network, num_samples=1000)

# Search for optimal sub-network
finder = CustomHardwareEvolutionFinder(
    ofa_network=ofa_network,
    latency_predictor=latency_predictor,
    accuracy_predictor=accuracy_predictor,
    latency_constraint=10.0,  # 10ms target latency
    quantization_scheme='int8'
)

best_config, best_metrics = finder.run_evolution_search()
print(f"Best accuracy: {best_metrics['accuracy']:.2f}%")
print(f"Latency: {best_metrics['latency']:.2f}ms")
```

### 3. Evaluate Quantized Model

```python
# Convert to ONNX and quantize
converter = OFAToONNXConverter(quantization_scheme='int8')
onnx_model = converter.convert_subnet(ofa_network, best_config)

# Evaluate on your dataset
from ofa.custom_hardware.evaluation import evaluate_quantized_model
accuracy = evaluate_quantized_model(
    onnx_model,
    dataset_path='/path/to/imagenet',
    providers=['YourNPUExecutionProvider']
)
```

## File Structure

```
ofa/
├── custom_hardware/
│   ├── __init__.py
│   ├── quantized_latency_predictor.py    # NPU latency prediction
│   ├── quantized_accuracy_predictor.py   # Quantized accuracy prediction  
│   ├── onnx_converter.py                 # OFA to ONNX conversion
│   ├── evolution_finder.py               # Hardware-specific NAS
│   ├── evaluation.py                     # Quantized model evaluation
│   └── utils.py                          # Utility functions
├── scripts/
│   ├── build_latency_lut.py             # Build latency lookup tables
│   ├── train_accuracy_predictor.py      # Train accuracy predictor
│   ├── search_custom_hardware.py        # Search optimal networks
│   └── evaluate_custom_hardware.py      # Evaluate on custom hardware
└── docs/
    └── custom_hardware_integration.md   # This file
```

## Detailed Workflow

### 1. Building Latency Lookup Tables

```bash
# Build latency lookup table for your NPU
python scripts/build_latency_lut.py \
    --hardware_name your_npu \
    --ofa_network ofa_mbv3_d234_e346_k357_w1.0 \
    --quantization_scheme int8 \
    --num_samples 2000 \
    --output_path latency_tables/your_npu_int8.yaml
```

### 2. Training Accuracy Predictors

```bash
# Train accuracy predictor for quantized models
python scripts/train_accuracy_predictor.py \
    --ofa_network ofa_mbv3_d234_e346_k357_w1.0 \
    --quantization_scheme int8 \
    --dataset_path /path/to/imagenet \
    --num_samples 5000 \
    --output_path accuracy_predictors/your_npu_int8.pth
```

### 3. Searching for Optimal Networks

```bash
# Search for optimal sub-networks
python scripts/search_custom_hardware.py \
    --hardware_name your_npu \
    --latency_constraint 10.0 \
    --quantization_scheme int8 \
    --population_size 200 \
    --max_iterations 1000 \
    --output_path results/your_npu_10ms.json
```

### 4. Evaluation

```bash
# Evaluate found sub-networks
python scripts/evaluate_custom_hardware.py \
    --config_path results/your_npu_10ms.json \
    --dataset_path /path/to/imagenet \
    --hardware_name your_npu \
    --quantization_scheme int8
```

## Configuration

### NPU Configuration

Create a configuration file for your NPU at `configs/your_npu.yaml`:

```yaml
hardware_name: "your_npu"
quantization_schemes:
  - int8
  - fp16
execution_providers:
  - "YourNPUExecutionProvider"
  - "CPUExecutionProvider"  # Fallback
batch_sizes: [1, 4, 8, 16]
input_shapes:
  - [224, 224]
  - [192, 192]
  - [160, 160]
optimization_level: "all"
memory_limit_mb: 2048
```

### Quantization Configuration

Define quantization settings in `configs/quantization_int8.yaml`:

```yaml
quantization_scheme: "int8"
calibration_dataset_size: 1000
activation_type: "QInt8"
weight_type: "QInt8"
per_channel: true
reduce_range: false
optimization_level: 99
```

## Advanced Features

### 1. Multi-Objective Optimization

```python
# Optimize for accuracy, latency, and power consumption
finder = CustomHardwareEvolutionFinder(
    objectives=['accuracy', 'latency', 'power'],
    constraints={
        'latency': 10.0,  # max 10ms
        'power': 500.0,  # max 500mW
    }
)
```

### 2. Custom Quantization Schemes

```python
# Define custom quantization
converter = OFAToONNXConverter(
    quantization_config={
        'activation_type': 'QInt8',
        'weight_type': 'QInt8', 
        'per_channel': True,
        'calibration_method': 'entropy'
    }
)
```

### 3. Hardware-Aware Layer Optimization

```python
# Optimize specific layer types for your NPU
latency_predictor = QuantizedLatencyPredictor(
    layer_optimizations={
        'conv2d': 'use_winograd',
        'depthwise_conv2d': 'channel_shuffle',
        'linear': 'quantized_gemm'
    }
)
```

## Troubleshooting

### Common Issues

1. **ONNX Runtime Provider Not Found**
   - Ensure your NPU's execution provider is properly installed
   - Check provider name spelling and availability

2. **Quantization Accuracy Drop**
   - Increase calibration dataset size
   - Try different quantization schemes (FP16 vs INT8)
   - Use per-channel quantization

3. **Latency Prediction Accuracy**
   - Build larger lookup tables with more samples
   - Include warm-up runs in latency measurements
   - Account for memory bandwidth limitations

### Performance Tips

1. **Batch Size Optimization**
   - Test different batch sizes for your hardware
   - Consider batch size in latency predictions

2. **Input Shape Optimization**
   - Profile different input resolutions
   - Use dynamic shape optimization where possible

3. **Memory Management**
   - Monitor memory usage during inference
   - Use memory pooling for better performance

## Contributing

To add support for new hardware:

1. Implement the execution provider interface
2. Add hardware configuration files
3. Test with standard OFA networks
4. Submit performance benchmarks

## References

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [OFA Paper](https://arxiv.org/abs/1908.09791)
- [Quantization Best Practices](https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model)
