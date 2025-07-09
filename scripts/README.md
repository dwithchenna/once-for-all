# OFA Custom Hardware Integration Scripts

This directory contains scripts for integrating custom hardware (NPU) support with the Once-for-All (OFA) framework.

## Scripts Overview

### Core Scripts

1. **`build_latency_lut.py`** - Build latency lookup tables for custom hardware
   - Samples various OFA subnet configurations
   - Converts them to quantized ONNX models  
   - Measures latency on target hardware
   - Creates lookup table for fast latency estimation during search

2. **`train_accuracy_predictor.py`** - Train quantized accuracy predictors
   - Generates training data by evaluating quantized subnets
   - Trains neural network to predict quantized accuracy
   - Accounts for quantization effects on model performance

3. **`search_custom_hardware.py`** - Search for optimal subnets
   - Uses evolutionary algorithm to find best architectures
   - Balances accuracy vs. latency for target hardware
   - Supports multi-objective optimization

4. **`evaluate_custom_hardware.py`** - Evaluate subnets on custom hardware
   - Measures actual accuracy and latency 
   - Compares PyTorch vs. quantized ONNX performance
   - Validates search results

### Example Workflow

5. **`../examples/custom_hardware_example.py`** - End-to-end example
   - Demonstrates complete workflow from start to finish
   - Includes quick demo mode for testing
   - Automatically runs all steps in sequence

## Quick Start

### 1. Prepare Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare configuration
cp configs/npu_config.yaml configs/my_npu_config.yaml
# Edit my_npu_config.yaml for your hardware
```

### 2. Run Complete Workflow

```bash
# Full workflow (may take several hours)
python examples/custom_hardware_example.py \
    --dataset_path /path/to/imagenet \
    --config configs/my_npu_config.yaml \
    --output_dir results/my_npu

# Quick demo (faster, for testing)
python examples/custom_hardware_example.py \
    --dataset_path /path/to/imagenet \
    --config configs/my_npu_config.yaml \
    --output_dir results/my_npu_demo \
    --quick_demo
```

### 3. Run Individual Steps

#### Build Latency Table
```bash
python scripts/build_latency_lut.py \
    --config configs/my_npu_config.yaml \
    --model_name ofa_mbv3_d234_e346_k357_w1.0 \
    --output_path latency_tables/my_npu_table.pkl \
    --num_samples 1000 \
    --verbose
```

#### Train Accuracy Predictor
```bash
python scripts/train_accuracy_predictor.py \
    --dataset_path /path/to/imagenet \
    --output_path predictors/my_npu_predictor.pth \
    --config configs/my_npu_config.yaml \
    --num_samples 2000 \
    --quantization \
    --verbose
```

#### Search Optimal Subnets
```bash
python scripts/search_custom_hardware.py \
    --hardware_name MyNPU \
    --latency_constraint 10.0 \
    --latency_table latency_tables/my_npu_table.pkl \
    --accuracy_predictor predictors/my_npu_predictor.pth \
    --output_path results/search_results.json \
    --verbose
```

#### Evaluate Selected Subnet
```bash
python scripts/evaluate_custom_hardware.py \
    --subnet_config results/best_subnet.json \
    --dataset_path /path/to/imagenet \
    --config configs/my_npu_config.yaml \
    --output_path results/evaluation.json \
    --save_onnx models/best_model.onnx \
    --quantization \
    --verbose
```

## Configuration

All scripts use YAML configuration files. Key sections:

- **`hardware`**: Execution provider and device settings
- **`quantization`**: Quantization parameters (INT8, FP16, etc.)
- **`search`**: Search algorithm parameters
- **`paths`**: Input/output directory settings

See `configs/npu_config.yaml` for detailed examples.

## Hardware Support

### Supported Providers
- **CPU**: `CPUExecutionProvider` (default, works everywhere)
- **CUDA GPU**: `CUDAExecutionProvider` 
- **OpenVINO**: `OpenVINOExecutionProvider` (Intel NPUs/VPUs)
- **Custom**: Any ONNX Runtime execution provider

### Adding New Hardware
1. Install appropriate ONNX Runtime provider package
2. Update configuration file with correct provider name
3. Set provider-specific options as needed
4. Test with a simple subnet evaluation

## Performance Tips

### For Faster Development
- Use `--quick_demo` mode for initial testing
- Reduce `num_samples` during development
- Use smaller datasets for accuracy predictor training
- Cache intermediate results when possible

### For Production
- Use sufficient samples for robust lookup tables (≥1000)
- Train accuracy predictors with diverse data (≥2000 samples)
- Run multiple search iterations for better results
- Validate results on full test sets

## Troubleshooting

### Common Issues

1. **ONNX Runtime Provider Not Found**
   ```
   Error: Provider 'XYZExecutionProvider' not available
   ```
   - Install correct ONNX Runtime package for your hardware
   - Check provider name spelling in config file

2. **Quantization Failures**
   ```
   Error: Quantization failed for model
   ```
   - Ensure calibration dataset is available
   - Check quantization type compatibility
   - Try dynamic quantization as fallback

3. **Poor Search Results**
   ```
   Warning: No valid architectures found under constraint
   ```
   - Increase latency constraint
   - Check that lookup table covers search space
   - Verify accuracy predictor is well-trained

4. **Memory Issues**
   ```
   Error: CUDA out of memory
   ```
   - Reduce batch sizes in config
   - Use CPU for accuracy predictor training
   - Process subnets sequentially instead of in batches

### Debug Mode
Add `--verbose` flag to any script for detailed logging:
```bash
python scripts/build_latency_lut.py --verbose [other args]
```

## Output Files

Each script generates specific output files:

- **Latency tables**: `.pkl` files with latency lookup data
- **Accuracy predictors**: `.pth` PyTorch model checkpoints  
- **Search results**: `.json` files with best architectures
- **Evaluation results**: `.json` files with performance metrics
- **ONNX models**: `.onnx` files ready for deployment

All outputs include metadata for reproducibility and debugging.

## Integration with Existing OFA Tools

These scripts are designed to work alongside existing OFA tools:

- Use same OFA networks from `ofa.model_zoo`
- Compatible with existing accuracy predictors
- Can use OFA's built-in subnet evaluation
- Outputs work with OFA's visualization tools
