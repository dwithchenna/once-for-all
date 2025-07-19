#!/usr/bin/env python3
"""
Script to save Pareto frontier models from collected dataset

This script identifies Pareto optimal models from the collected dataset
and saves them as ONNX models with accuracy and latency information in the filename.
"""

import os
import sys
import json
import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import torch

# Add OFA modules to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing OFA modules
try:
    from ofa.model_zoo import ofa_net
    OFA_AVAILABLE = True
except ImportError:
    OFA_AVAILABLE = False
    print("Warning: OFA modules not available.")

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_static, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX/ONNX Runtime not available. Please install: pip install onnx onnxruntime")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleONNXConverter:
    """Simple ONNX converter with static quantization."""
    
    def __init__(self, quantization_enabled=True, weight_type="QInt8", activation_type="QUInt8"):
        self.quantization_enabled = quantization_enabled
        self.weight_type = weight_type
        self.activation_type = activation_type
    
    def _create_calibration_data(self, input_shape, num_samples=50):
        """Create calibration data for static quantization."""
        calibration_data = []
        for i in range(num_samples):
            # Create random input data
            data = np.random.randn(*input_shape).astype(np.float32)
            calibration_data.append(data)
        return calibration_data
    
    def convert_subnet(self, subnet, input_shape, output_path):
        """Convert PyTorch model to ONNX with static quantization."""
        try:
            if not ONNX_AVAILABLE:
                logger.warning("ONNX conversion skipped - missing dependencies")
                return output_path
                
            # Create dummy input
            dummy_input = torch.randn(*input_shape)
            
            # Create directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert to ONNX
            torch.onnx.export(
                subnet,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=11,
                do_constant_folding=True
            )
            
            # Apply static quantization if enabled
            if self.quantization_enabled and ONNX_AVAILABLE:
                quantized_path = output_path.replace(".onnx", "_quantized.onnx")
                try:
                    from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
                    
                    # Create calibration data reader
                    class CalibrationDataReader(CalibrationDataReader):
                        def __init__(self, calibration_data):
                            self.calibration_data = calibration_data
                            self.data_index = 0
                        
                        def get_next(self):
                            if self.data_index >= len(self.calibration_data):
                                return None
                            data = {'input': self.calibration_data[self.data_index]}
                            self.data_index += 1
                            return data
                    
                    # Create calibration data
                    calibration_data = self._create_calibration_data(input_shape)
                    dr = CalibrationDataReader(calibration_data)
                    
                    # Apply static quantization
                    quantize_static(
                        output_path,
                        quantized_path,
                        dr,
                        weight_type=QuantType.QInt8,
                        activation_type=QuantType.QUInt8
                    )
                    
                    logger.info(f"Static quantization applied: {quantized_path}")
                    
                    # Remove original and return quantized path
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    return quantized_path
                    
                except Exception as e:
                    logger.warning(f"Static quantization failed: {e}")
                    return output_path
            
            return output_path
            
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            return output_path


class ParetoModelSaver:
    """Save Pareto frontier models from collected dataset."""
    
    def __init__(self, data_path, output_dir="./data/collected/onnx_models/pareto"):
        """Initialize with dataset path and output directory."""
        self.data_path = data_path
        self.output_dir = output_dir
        self.data = self._load_data()
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process dataset
        self._process_data()
        
        # Initialize ONNX converter
        self.onnx_converter = SimpleONNXConverter(
            quantization_enabled=True,
            weight_type="QInt8",
            activation_type="QUInt8"
        )
        
        # Initialize OFA network
        if OFA_AVAILABLE:
            try:
                self.ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
                logger.info("OFA network loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load OFA network: {e}")
                self.ofa_network = None
        else:
            self.ofa_network = None
    
    def _load_data(self):
        """Load dataset from JSON file."""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} data points from {self.data_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []
    
    def _process_data(self):
        """Process data to extract required information."""
        # Extract accuracy and latency
        self.accuracies = np.array([item.get('accuracy', 0) for item in self.data])
        self.latencies = np.array([item.get('latency', 0) for item in self.data])
        self.configs = [item.get('config', {}) for item in self.data]
        
        # Filter out invalid values
        valid_indices = np.where(np.isfinite(self.accuracies) & np.isfinite(self.latencies))[0]
        self.accuracies = self.accuracies[valid_indices]
        self.latencies = self.latencies[valid_indices]
        self.configs = [self.configs[i] for i in valid_indices]
        self.valid_indices = valid_indices
        
        logger.info(f"Found {len(valid_indices)} valid data points")
    
    def _estimate_flops(self, config, resolution):
        """Estimate FLOPs (in millions) for a subnet configuration.
        
        This is a simplified calculation based on the configuration.
        For MobileNetV3-like architectures, FLOPs are primarily determined by:
        1. Input resolution (squared impact)
        2. Expansion ratios
        3. Kernel sizes
        4. Depths of each stage
        """
        # Base FLOPs for OFA MobileNetV3 with default settings at 224x224
        base_flops = 300  # ~300M FLOPs for default configuration
        
        # Resolution impact (squared)
        resolution_factor = (resolution / 224) ** 2
        
        # Configuration impact
        ks_factor = np.mean(config.get('ks', [3] * 20)) / 3.0  # Normalized by base kernel size 3
        e_factor = np.mean(config.get('e', [3] * 20)) / 3.0    # Normalized by base expansion 3
        d_factor = sum(config.get('d', [2] * 5)) / 10.0        # Normalized by base depth sum 10
        
        # Combine factors (weighted importance)
        config_factor = 0.2 * ks_factor + 0.5 * e_factor + 0.3 * d_factor
        
        # Calculate final FLOPs estimation
        flops = base_flops * resolution_factor * config_factor
        
        return flops
    
    def _estimate_params(self, config):
        """Estimate parameters (in millions) for a subnet configuration.
        
        Parameters in MobileNetV3-like architectures are mainly affected by:
        1. Expansion ratios
        2. Kernel sizes
        3. Depths of each stage
        Resolution doesn't affect parameter count.
        """
        # Base parameters for OFA MobileNetV3 with default settings
        base_params = 5.0  # ~5M parameters for default configuration
        
        # Configuration impact
        ks_factor = np.mean(config.get('ks', [3] * 20)) / 3.0  # Kernel size impact
        e_factor = np.mean(config.get('e', [3] * 20)) / 3.0    # Expansion ratio impact
        d_factor = sum(config.get('d', [2] * 5)) / 10.0        # Depth impact
        
        # Combine factors (weighted importance)
        # Expansion ratio and depth have larger impact on parameter count than kernel size
        config_factor = 0.1 * ks_factor + 0.6 * e_factor + 0.3 * d_factor
        
        # Calculate final parameter estimation
        params = base_params * config_factor
        
        return params
    
    def _find_pareto_frontier(self):
        """Find Pareto frontier points (maximize accuracy, minimize latency)."""
        pareto_indices = []
        
        for i in range(len(self.accuracies)):
            is_pareto = True
            
            for j in range(len(self.accuracies)):
                if i != j:
                    # Point j dominates point i if it has better accuracy and better/equal latency
                    if (self.accuracies[j] >= self.accuracies[i] and 
                        self.latencies[j] <= self.latencies[i] and
                        (self.accuracies[j] > self.accuracies[i] or self.latencies[j] < self.latencies[i])):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return np.array(pareto_indices)
    
    def save_pareto_models(self):
        """Save all models on the Pareto frontier."""
        if not OFA_AVAILABLE or self.ofa_network is None:
            logger.error("OFA network not available, cannot save models")
            return
        
        # Find Pareto frontier
        pareto_indices = self._find_pareto_frontier()
        logger.info(f"Found {len(pareto_indices)} models on the Pareto frontier")
        
        if len(pareto_indices) == 0:
            logger.warning("No Pareto frontier models found")
            return
        
        # Sort by latency for easier reference
        sort_idx = np.argsort(self.latencies[pareto_indices])
        pareto_indices = pareto_indices[sort_idx]
        
        # Save each model
        saved_count = 0
        for i, idx in enumerate(tqdm(pareto_indices, desc="Saving Pareto models")):
            try:
                # Get subnet config
                config = self.configs[idx]
                accuracy = self.accuracies[idx]
                latency = self.latencies[idx]
                
                # Set resolution
                resolution = config.get('r', 224)
                input_shape = (1, 3, resolution, resolution)
                
                # Calculate model complexity (approximate FLOPs and params)
                ks_avg = np.mean(config.get('ks', [3] * 20))
                e_avg = np.mean(config.get('e', [3] * 20))
                d_sum = sum(config.get('d', [2] * 5))
                
                # Estimate FLOPs (in millions) - simplified calculation
                flops = self._estimate_flops(config, resolution)
                
                # Estimate parameters (in millions) - simplified calculation
                params = self._estimate_params(config)
                
                # Format filename with detailed information
                filename = (f"pareto_{i+1:02d}_acc{accuracy:.3f}_lat{latency:.1f}ms_"
                           f"r{resolution}_d{d_sum}_ks{ks_avg:.1f}_e{e_avg:.1f}_"
                           f"flops{flops:.1f}M_params{params:.1f}M.onnx")
                onnx_path = os.path.join(self.output_dir, filename)
                
                # Set active subnet
                self.ofa_network.set_active_subnet(
                    ks=config.get('ks', [3] * 20),
                    e=config.get('e', [3] * 20),
                    d=config.get('d', [2] * 5)
                )
                subnet = self.ofa_network.get_active_subnet(preserve_weight=True)
                
                # Convert to ONNX
                onnx_path = self.onnx_converter.convert_subnet(subnet, input_shape, onnx_path)
                
                logger.info(f"Saved Pareto model {i+1}/{len(pareto_indices)}: acc={accuracy:.3f}, lat={latency:.1f}ms")
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Error saving Pareto model {i+1}/{len(pareto_indices)}: {e}")
        
        logger.info(f"Successfully saved {saved_count} out of {len(pareto_indices)} Pareto frontier models")
        logger.info(f"Models saved to: {self.output_dir}")


def main():
    """Main function to run Pareto model saver."""
    parser = argparse.ArgumentParser(description="Save Pareto frontier models from collected dataset")
    parser.add_argument("--data", required=True, help="Path to collected dataset JSON file")
    parser.add_argument("--output", default="./data/collected/onnx_models/pareto", 
                        help="Output directory for Pareto models")
    args = parser.parse_args()
    
    # Initialize and run model saver
    saver = ParetoModelSaver(args.data, args.output)
    saver.save_pareto_models()


if __name__ == "__main__":
    main()
