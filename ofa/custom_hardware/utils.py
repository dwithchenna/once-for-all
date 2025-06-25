"""
Utility functions for custom hardware integration
"""

import os
import yaml
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def load_hardware_config(config_path: str) -> Dict[str, Any]:
    """
    Load hardware configuration from YAML file.
    
    Args:
        config_path: Path to hardware configuration file
        
    Returns:
        Hardware configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load hardware config from {config_path}: {e}")
        return {}


def save_results(results: Dict[str, Any], output_path: str):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Path to save results
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert object to JSON serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


def create_test_input(input_shape: Tuple[int, int, int, int], batch_size: int = 1) -> np.ndarray:
    """
    Create test input tensor for model inference.
    
    Args:
        input_shape: Input tensor shape (B, C, H, W)
        batch_size: Batch size override
        
    Returns:
        Random test input tensor
    """
    # Override batch size if specified
    shape = (batch_size, input_shape[1], input_shape[2], input_shape[3])
    
    # Create random input in range [0, 1]
    test_input = np.random.rand(*shape).astype(np.float32)
    
    # Normalize to ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    test_input = (test_input - mean) / std
    
    return test_input


def validate_subnet_config(config: Dict[str, Any]) -> bool:
    """
    Validate subnet configuration dictionary.
    
    Args:
        config: Subnet configuration
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['ks', 'e', 'd']
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required key in subnet config: {key}")
            return False
    
    # Validate lengths
    if len(config.get('ks', [])) != 20:
        logger.error(f"Invalid ks length: {len(config.get('ks', []))}, expected 20")
        return False
    
    if len(config.get('e', [])) != 20:
        logger.error(f"Invalid e length: {len(config.get('e', []))}, expected 20")
        return False
    
    if len(config.get('d', [])) != 5:
        logger.error(f"Invalid d length: {len(config.get('d', []))}, expected 5")
        return False
    
    # Validate value ranges
    valid_ks = [3, 5, 7]
    valid_e = [3, 4, 6]
    valid_d = [2, 3, 4]
    
    for ks in config['ks']:
        if ks not in valid_ks:
            logger.error(f"Invalid kernel size: {ks}, must be one of {valid_ks}")
            return False
    
    for e in config['e']:
        if e not in valid_e:
            logger.error(f"Invalid expand ratio: {e}, must be one of {valid_e}")
            return False
    
    for d in config['d']:
        if d not in valid_d:
            logger.error(f"Invalid depth: {d}, must be one of {valid_d}")
            return False
    
    return True


def format_subnet_config(config: Dict[str, Any]) -> str:
    """
    Format subnet configuration as readable string.
    
    Args:
        config: Subnet configuration
        
    Returns:
        Formatted configuration string
    """
    parts = []
    
    if 'ks' in config:
        ks_str = '-'.join(map(str, config['ks']))
        parts.append(f"ks={ks_str}")
    
    if 'e' in config:
        e_str = '-'.join(map(str, config['e']))
        parts.append(f"e={e_str}")
    
    if 'd' in config:
        d_str = '-'.join(map(str, config['d']))
        parts.append(f"d={d_str}")
    
    if 'r' in config:
        r_str = str(config['r'][0]) if isinstance(config['r'], list) else str(config['r'])
        parts.append(f"r={r_str}")
    
    return "_".join(parts)


def calculate_model_complexity(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate approximate model complexity metrics.
    
    Args:
        config: Subnet configuration
        
    Returns:
        Dictionary with complexity metrics
    """
    # Simplified complexity calculation
    # In practice, this should be more sophisticated
    
    ks = config.get('ks', [7] * 20)
    e = config.get('e', [6] * 20)
    d = config.get('d', [4] * 5)
    r = config.get('r', [224])[0] if 'r' in config else 224
    
    # Approximate parameter count
    param_count = 0
    for k, expand in zip(ks, e):
        # Simplified parameter estimation for MBConv blocks
        param_count += k * k * expand * 32  # Rough approximation
    
    # Add depth contribution
    depth_factor = sum(d) / len(d)
    param_count *= depth_factor
    
    # Approximate FLOPs
    flops = param_count * r * r / 1000000  # Rough FLOP estimation in MFLOPs
    
    # Model size estimate (MB)
    model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming FP32
    
    return {
        'parameter_count': param_count,
        'flops_mflops': flops,
        'model_size_mb': model_size_mb,
        'depth_factor': depth_factor,
        'avg_kernel_size': np.mean(ks),
        'avg_expand_ratio': np.mean(e)
    }


def create_hardware_config_template(hardware_name: str, output_path: str):
    """
    Create a template hardware configuration file.
    
    Args:
        hardware_name: Name of the hardware
        output_path: Path to save the template
    """
    template = {
        'hardware_name': hardware_name,
        'quantization_schemes': ['int8', 'fp16'],
        'execution_providers': [
            f'{hardware_name}ExecutionProvider',
            'CPUExecutionProvider'  # Fallback
        ],
        'batch_sizes': [1, 4, 8, 16],
        'input_shapes': [
            [160, 160],
            [192, 192], 
            [224, 224]
        ],
        'warmup_runs': 10,
        'measurement_runs': 50,
        'optimization_level': 'all',
        'memory_limit_mb': 2048,
        'max_threads': 4,
        'enable_profiling': False,
        'custom_options': {
            'example_option_1': 'value1',
            'example_option_2': 42
        }
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Hardware config template created: {output_path}")


def load_calibration_dataset(dataset_path: str, num_samples: int = 100) -> List[np.ndarray]:
    """
    Load calibration dataset for quantization.
    
    Args:
        dataset_path: Path to calibration dataset
        num_samples: Number of samples to load
        
    Returns:
        List of calibration samples
    """
    # Placeholder implementation - replace with actual dataset loading
    # For now, return random data
    samples = []
    for _ in range(num_samples):
        sample = np.random.rand(3, 224, 224).astype(np.float32)
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        sample = (sample - mean) / std
        samples.append(sample)
    
    logger.warning(f"Using random calibration data. Replace with real dataset loading.")
    return samples


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def generate_subnet_configs(
    num_configs: int,
    ks_choices: List[int] = [3, 5, 7],
    e_choices: List[int] = [3, 4, 6],
    d_choices: List[int] = [2, 3, 4],
    r_choices: List[int] = [160, 192, 224]
) -> List[Dict[str, Any]]:
    """
    Generate random subnet configurations for testing.
    
    Args:
        num_configs: Number of configurations to generate
        ks_choices: Available kernel size choices
        e_choices: Available expand ratio choices
        d_choices: Available depth choices
        r_choices: Available resolution choices
        
    Returns:
        List of subnet configurations
    """
    configs = []
    
    for _ in range(num_configs):
        config = {
            'ks': [np.random.choice(ks_choices) for _ in range(20)],
            'e': [np.random.choice(e_choices) for _ in range(20)],
            'd': [np.random.choice(d_choices) for _ in range(5)],
            'r': [np.random.choice(r_choices)]
        }
        configs.append(config)
    
    return configs


def compare_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two subnet configurations.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        Comparison results
    """
    comparison = {
        'identical': config1 == config2,
        'differences': {}
    }
    
    for key in ['ks', 'e', 'd', 'r']:
        if key in config1 and key in config2:
            if config1[key] != config2[key]:
                comparison['differences'][key] = {
                    'config1': config1[key],
                    'config2': config2[key]
                }
        elif key in config1 or key in config2:
            comparison['differences'][key] = {
                'config1': config1.get(key, 'missing'),
                'config2': config2.get(key, 'missing')
            }
    
    return comparison
