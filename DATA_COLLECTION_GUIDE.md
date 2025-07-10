# Step-by-Step Guide: Collecting Data for Quantized Predictors

This guide will help you collect accuracy and latency data on your custom NPU hardware for training quantized predictors used in Neural Architecture Search (NAS).

## Prerequisites

1. **Hardware Setup**: Ensure your custom NPU hardware is properly configured
2. **ONNX Runtime Provider**: Install the appropriate ONNX Runtime execution provider for your NPU
3. **Python Environment**: Python 3.8+ with pip installed

## Step 1: Install Dependencies

Run the installation script:
```bash
install_deps.bat
```

Or install manually:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime PyYAML tqdm numpy filelock gdown scikit-learn matplotlib
```

## Step 2: Configure Hardware Settings

Edit `configs/stx_npu_config.yaml` to match your hardware:

```yaml
hardware:
  name: "YourCustomNPU"
  provider: "YourNPUProvider"  # e.g., "OpenVINOExecutionProvider"
  provider_options:
    device_type: "NPU"
    precision: "FP16"
```

Common providers:
- `CPUExecutionProvider` (default/fallback)
- `OpenVINOExecutionProvider` (Intel NPU)
- `CUDAExecutionProvider` (NVIDIA GPU)
- `QNNExecutionProvider` (Qualcomm NPU)

## Step 3: Verify Setup

Test your configuration:
```bash
python test_setup.py
```

This will verify:
- ✓ All packages are installed
- ✓ Configuration file is valid
- ✓ Directories are created
- ✓ OFA models can be loaded

## Step 4: Collect Data

### Important: Quantized Accuracy Measurement

The data collection script measures accuracy on **quantized ONNX models**, not PyTorch models. This ensures the collected data reflects real hardware performance:

1. **Subnet → Quantized ONNX**: Each subnet is first converted to quantized ONNX format
2. **Accuracy on Quantized Model**: Accuracy is measured using the quantized ONNX model
3. **Latency on Same Model**: Latency is measured on the same quantized ONNX model
4. **Quantization Impact**: The data includes quantization degradation effects (typically 1-4% accuracy loss)

### Option A: Quick Test (100 samples)
```bash
python collect_data.py --config configs/stx_npu_config.yaml --samples 100 --output data/collected/test_dataset.json
```

### Option B: Full Dataset (1000+ samples)
```bash
python collect_data.py --config configs/stx_npu_config.yaml --samples 1000 --output data/collected/full_dataset.json
```

### Option C: Large Dataset (5000+ samples for better accuracy)
```bash
python collect_data.py --config configs/stx_npu_config.yaml --samples 5000 --output data/collected/large_dataset.json
```

## Step 5: Monitor Data Collection

The script will:
1. **Generate subnet configurations** - Random and systematic sampling
2. **Measure accuracy** - Evaluate each subnet configuration
3. **Convert to ONNX** - Apply quantization settings
4. **Measure latency** - Run on your custom hardware
5. **Save data points** - Checkpoint every 100 samples

Progress will be shown with:
```
Collecting data: 45%|████▌     | 450/1000 [02:30<02:45, 3.33it/s]
```

## Step 6: Train Predictors

Once data collection is complete:

### Train both predictors:
```bash
python train_predictors.py --dataset data/collected/complete_dataset.json --epochs 100
```

### Train with custom settings:
```bash
python train_predictors.py --dataset data/collected/complete_dataset.json --epochs 200 --batch_size 64
```

## Step 7: Verify Results

After training, you'll find:
- `models/accuracy_predictor.pth` - Trained accuracy predictor
- `models/latency_predictor.pth` - Trained latency predictor
- `plots/accuracy_predictor_results.png` - Training curves and validation plots
- `plots/latency_predictor_results.png` - Training curves and validation plots

## Step 8: Visualize Data

Create comprehensive visualizations of your collected data:

### Quick Demo (with sample data):
```bash
python demo_visualization.py
```

### Full Visualization:
```bash
python visualize_data.py --data data/collected/complete_dataset.json --output plots
```

### Summary Statistics Only:
```bash
python visualize_data.py --data data/collected/complete_dataset.json --summary
```

The visualization script creates:
- **Accuracy vs Latency Plot**: Main trade-off visualization with circle size = model complexity (FLOPs)
- **Efficiency Frontier**: Shows the most efficient models (best accuracy/latency ratio)
- **Configuration Analysis**: How different parameters affect performance
- **Summary Statistics**: Distributions and correlations of all metrics

### Example Output:
```
OFA DATA COLLECTION SUMMARY
============================================================
Total data points: 1000
Valid data points: 987

PERFORMANCE METRICS:
  Accuracy: 0.764 ± 0.042
  Latency:  18.3 ± 12.7 ms
  Model Size: 4.2 ± 1.8 M params
  FLOPs:    156.4 ± 89.2 M FLOPs

PARETO FRONTIER: 23 models
  Best Accuracy: 0.847 @ 28.4ms
  Best Latency:  5.2ms @ 0.698 acc
  Best Efficiency: 0.0421 @ 0.798 acc, 18.9ms
```

## Expected Timeline

| Task | Samples | Time Estimate |
|------|---------|---------------|
| Setup & Test | - | 10-15 minutes |
| Data Collection | 100 | 30-45 minutes |
| Data Collection | 1000 | 3-5 hours |
| Data Collection | 5000 | 15-20 hours |
| Predictor Training | - | 10-30 minutes |
| Data Visualization | - | 2-5 minutes |

## Configuration Tips

### For Faster Collection:
```yaml
performance:
  warmup_runs: 5      # Reduce from 10
  measurement_runs: 50 # Reduce from 100
  eval_batch_size: 100 # Increase from 50
```

### For Better Accuracy:
```yaml
performance:
  warmup_runs: 20     # Increase from 10
  measurement_runs: 200 # Increase from 100
  max_eval_batches: 200 # Increase from 100
```

### For Your Custom NPU:
```yaml
hardware:
  name: "CustomNPU"
  provider: "YourProvider"
  provider_options:
    device_type: "NPU"
    precision: "INT8"  # or "FP16"
    num_threads: 4
```

## Troubleshooting

### Common Issues:

1. **ONNX Runtime Provider Not Found**
   - Install the correct ONNX Runtime package for your hardware
   - Check provider name spelling in config

2. **Quantization Errors**
   - Try `mode: "dynamic"` instead of `"static"`
   - Reduce `calibration_dataset_size`

3. **Memory Issues**
   - Reduce `batch_size` in training
   - Reduce `eval_batch_size` in config

4. **Slow Performance**
   - Reduce `measurement_runs` and `warmup_runs`
   - Enable `batch_evaluation: true`

5. **JSON Serialization Errors**
   - If you see "Object of type int64/float32 is not JSON serializable"
   - This happens when NumPy types aren't properly converted to Python types
   - The script includes a custom JSON encoder to handle this automatically

6. **Infinity or NaN Values**
   - The data collection script now automatically detects and fixes infinity/NaN values
   - These can occur when measuring latency on unsupported models or hardware configurations
   - Fixed values are replaced with reasonable defaults (25ms for latency, 0.75 for accuracy)
   - A warning message will be logged when values are replaced

7. **Training with Small Datasets**
   - When training with very small datasets (<100 samples), you might encounter NaN errors
   - The training script is robust to handle small datasets and will filter out invalid values
   - For best results, use at least 1000 samples for a reliable predictor

### Getting Help:

Check logs for detailed error messages:
```bash
python collect_data.py --config configs/stx_npu_config.yaml --samples 10 > collection.log 2>&1
```

## Using the Trained Predictors

Once trained, you can use the predictors in your NAS pipeline:

```python
# Load trained models
accuracy_predictor = QuantizedAccuracyPredictor('models/accuracy_predictor.pth')
latency_predictor = QuantizedLatencyPredictor('models/latency_predictor.pth')

# Use in evolution search
from ofa.custom_hardware import CustomHardwareEvolutionFinder

finder = CustomHardwareEvolutionFinder(
    accuracy_predictor=accuracy_predictor,
    latency_predictor=latency_predictor
)

# Find optimal subnet
best_subnet = finder.run_evolution_search(
    constraint=30.0,  # 30ms latency constraint
    verbose=True
)
```

## Next Steps

1. **Validate Predictors**: Test predictions against real measurements
2. **Collect More Data**: Continuously improve predictors with more samples
3. **Hardware-Specific Tuning**: Fine-tune for your specific NPU characteristics
4. **Integration**: Integrate predictors into your NAS workflow
5. **Visualization Analysis**: Use plots to understand performance trade-offs

## Summary

You now have a complete pipeline for:
- ✅ **Data Collection**: Automated collection of accuracy and latency data
- ✅ **Predictor Training**: Neural networks that predict performance without actual deployment
- ✅ **Visualization**: Comprehensive plots showing performance trade-offs
- ✅ **Hardware Integration**: Support for custom NPU hardware with quantization

The visualization script provides insights into:
- **Pareto Frontier**: Optimal trade-off between accuracy and latency
- **Model Complexity**: How FLOPs and parameters affect performance
- **Configuration Impact**: Which architectural choices matter most
- **Hardware Efficiency**: Best models for your specific constraints

Good luck with your Neural Architecture Search!
