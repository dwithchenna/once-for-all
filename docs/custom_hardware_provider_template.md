# Custom Hardware Provider Template

This template shows how to add support for a new hardware accelerator
to the OFA custom hardware integration framework.

## Hardware Provider Configuration

Create a new configuration file based on your hardware:

```yaml
# configs/my_hardware_config.yaml

hardware:
  name: "MyCustomHardware"
  provider: "MyCustomExecutionProvider"  # Your ONNX Runtime provider
  provider_options:
    device_id: 0
    # Add hardware-specific options here
    
quantization:
  enabled: true
  weight_type: "QInt8"
  activation_type: "QUInt8"
  
  # Hardware-specific quantization settings
  per_channel: true
  reduce_range: false
  
search:
  efficiency_constraint: 5.0  # Adjust based on your hardware capabilities
  accuracy_threshold: 0.75
```

## ONNX Runtime Provider Setup

### Option 1: Use Existing Provider
If your hardware is supported by an existing ONNX Runtime provider:

```python
# Example for Intel VPU/NPU via OpenVINO
config = {
    'hardware': {
        'provider': 'OpenVINOExecutionProvider',
        'provider_options': {
            'device_type': 'NPU',
            'precision': 'FP16'
        }
    }
}
```

### Option 2: Custom Provider
If you need a custom ONNX Runtime provider:

1. Implement your provider following ONNX Runtime guidelines
2. Install your provider package
3. Configure it in the hardware config

```python
# Example custom provider usage
config = {
    'hardware': {
        'provider': 'MyCustomProvider',
        'provider_options': {
            'device_path': '/dev/my_accelerator',
            'compute_precision': 'INT8'
        }
    }
}
```

## Hardware-Specific Optimizations

### Custom ONNX Optimization Passes

If your hardware benefits from specific ONNX graph optimizations:

```python
# ofa/custom_hardware/my_hardware_optimizer.py

import onnx
from onnx import helper
from typing import Optional

class MyHardwareOptimizer:
    """Custom ONNX optimizer for MyHardware"""
    
    def __init__(self, config: dict):
        self.config = config
    
    def optimize_model(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Apply hardware-specific optimizations"""
        
        # Example: Fuse specific operators for your hardware
        optimized_model = self._fuse_custom_ops(model)
        
        # Example: Quantize specific layers differently
        optimized_model = self._apply_custom_quantization(optimized_model)
        
        return optimized_model
    
    def _fuse_custom_ops(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse operators for better hardware utilization"""
        # Implement your operator fusion logic
        return model
    
    def _apply_custom_quantization(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Apply hardware-specific quantization"""
        # Implement your quantization logic
        return model

# Usage in ONNXConverter
from ofa.custom_hardware import ONNXConverter

class MyHardwareConverter(ONNXConverter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = MyHardwareOptimizer(self.config)
    
    def _optimize_model(self, model: onnx.ModelProto) -> onnx.ModelProto:
        # Apply base optimizations
        model = super()._optimize_model(model)
        
        # Apply hardware-specific optimizations
        model = self.optimizer.optimize_model(model)
        
        return model
```

### Custom Latency Measurement

For more accurate latency measurement on your hardware:

```python
# ofa/custom_hardware/my_hardware_latency.py

import time
import numpy as np
from typing import List, Dict

class MyHardwareLatencyMeasurer:
    """Hardware-specific latency measurement"""
    
    def __init__(self, provider: str, provider_options: dict):
        self.provider = provider
        self.provider_options = provider_options
    
    def measure_latency(self, onnx_model_path: str, 
                       input_shape: tuple,
                       num_runs: int = 100) -> Dict[str, float]:
        """Measure latency with hardware-specific timing"""
        
        # Initialize your hardware session
        session = self._create_session(onnx_model_path)
        
        # Hardware-specific warm-up
        self._hardware_warmup(session, input_shape)
        
        # Measure with hardware-specific timing
        latencies = []
        for _ in range(num_runs):
            latency = self._measure_single_run(session, input_shape)
            latencies.append(latency)
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99)
        }
    
    def _create_session(self, onnx_model_path: str):
        """Create hardware-specific ONNX Runtime session"""
        import onnxruntime as ort
        
        # Add hardware-specific session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        return ort.InferenceSession(
            onnx_model_path,
            session_options,
            providers=[(self.provider, self.provider_options)]
        )
    
    def _hardware_warmup(self, session, input_shape: tuple):
        """Hardware-specific warmup procedure"""
        # Implement warmup logic for your hardware
        pass
    
    def _measure_single_run(self, session, input_shape: tuple) -> float:
        """Measure single inference with hardware timing"""
        # Use hardware-specific timing mechanisms
        # E.g., GPU events, NPU performance counters, etc.
        import torch
        
        dummy_input = torch.randn(input_shape).numpy()
        
        # Hardware-specific timing start
        start_time = time.perf_counter()
        
        # Run inference
        _ = session.run(None, {'input': dummy_input})
        
        # Hardware-specific timing end  
        end_time = time.perf_counter()
        
        return (end_time - start_time) * 1000  # Convert to ms
```

## Testing Your Hardware Integration

### 1. Test Basic Functionality
```bash
# Test that your provider works
python test_integration.py

# Test with your configuration
python examples/quick_usage_example.py
```

### 2. Test Latency Measurement
```bash
# Build latency table for your hardware
python scripts/build_latency_lut.py \
    --config configs/my_hardware_config.yaml \
    --num_samples 100 \
    --verbose
```

### 3. Test Accuracy Prediction
```bash
# Train accuracy predictor (requires ImageNet)
python scripts/train_accuracy_predictor.py \
    --dataset_path /path/to/imagenet \
    --config configs/my_hardware_config.yaml \
    --num_samples 500 \
    --quantization \
    --verbose
```

### 4. Test Complete Workflow
```bash
# Run end-to-end workflow
python examples/custom_hardware_example.py \
    --dataset_path /path/to/imagenet \
    --config configs/my_hardware_config.yaml \
    --quick_demo \
    --verbose
```

## Performance Tuning

### Optimize for Your Hardware
1. **Quantization Settings**: Experiment with different quantization schemes
2. **Batch Size**: Find optimal batch size for your hardware
3. **Graph Optimizations**: Enable hardware-specific ONNX optimizations
4. **Memory Management**: Configure memory allocation strategies

### Example Optimized Configuration
```yaml
hardware:
  name: "OptimizedMyHardware"
  provider: "MyCustomProvider"
  provider_options:
    device_id: 0
    memory_limit: "4GB"
    precision: "INT8"
    enable_fast_math: true
    
quantization:
  enabled: true
  weight_type: "QInt8"
  activation_type: "QUInt8"
  per_channel: true
  reduce_range: false
  
  # Hardware-specific optimizations
  use_qdq_ops: true
  quantization_mode: "static"
  
performance:
  warmup_runs: 20  # Increase for stable measurements
  measurement_runs: 200
  enable_profiling: true
  
  # Batch optimization
  batch_evaluation: true
  optimal_batch_size: 1
```

## Debugging Common Issues

### Provider Not Found
```
Error: Provider 'MyCustomProvider' not available
```
- Ensure provider package is installed
- Check provider name spelling
- Verify hardware drivers are installed

### Quantization Failures
```
Error: Quantization failed for model
```
- Check quantization type compatibility
- Ensure calibration data is available
- Try dynamic quantization as fallback

### Poor Performance
```
Warning: Latency measurements are inconsistent
```
- Increase warmup runs
- Check for thermal throttling
- Verify hardware is not being shared
- Use hardware-specific timing mechanisms

### Memory Issues
```
Error: Insufficient memory for model
```
- Reduce batch size
- Use external data format for large models
- Configure provider memory limits

## Contributing Your Hardware Support

To contribute your hardware integration back to OFA:

1. **Document Your Hardware**: Create comprehensive setup instructions
2. **Add Tests**: Include hardware-specific test cases
3. **Configuration Templates**: Provide working config examples
4. **Performance Benchmarks**: Share performance results
5. **Submit PR**: Include all new files and documentation

### Example Contribution Structure
```
ofa/custom_hardware/providers/
├── my_hardware/
│   ├── __init__.py
│   ├── optimizer.py
│   ├── latency_measurer.py
│   └── config_template.yaml
configs/
├── my_hardware_config.yaml
docs/
├── my_hardware_setup.md
tests/
├── test_my_hardware.py
```

This template provides a starting point for integrating any custom hardware
with the OFA framework while maintaining compatibility with the existing API.
