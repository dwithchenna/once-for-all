"""
Quantized Latency Predictor for Custom NPU Hardware

This module provides latency prediction for quantized ONNX models running
on custom NPU hardware using ONNX Runtime.
"""

import os
import time
import yaml
import numpy as np
import onnx
import onnxruntime as ort
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging

from ofa.utils import val2list
from .utils import load_hardware_config, create_test_input
from .onnx_converter import OFAToONNXConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantizedLatencyPredictor:
    """
    Predicts latency of quantized OFA sub-networks on custom NPU hardware.
    
    This class builds lookup tables by actually running quantized ONNX models
    on the target hardware and measuring real latency.
    """
    
    def __init__(
        self,
        hardware_name: str,
        quantization_scheme: str = 'int8',
        onnx_runtime_providers: Optional[List[str]] = None,
        config_path: Optional[str] = None,
        cache_dir: str = './latency_cache'
    ):
        """
        Initialize the latency predictor.
        
        Args:
            hardware_name: Name of the target hardware (e.g., 'custom_npu')
            quantization_scheme: Quantization scheme ('int8', 'fp16', etc.)
            onnx_runtime_providers: List of ONNX Runtime execution providers
            config_path: Path to hardware configuration file
            cache_dir: Directory to cache latency measurements
        """
        self.hardware_name = hardware_name
        self.quantization_scheme = quantization_scheme
        self.cache_dir = cache_dir
        self.lookup_table = defaultdict(dict)
        
        # Load hardware configuration
        if config_path:
            self.config = load_hardware_config(config_path)
        else:
            self.config = self._get_default_config()
            
        # Setup ONNX Runtime providers
        if onnx_runtime_providers:
            self.providers = onnx_runtime_providers
        else:
            self.providers = self.config.get('execution_providers', ['CPUExecutionProvider'])
            
        # Initialize ONNX converter
        self.onnx_converter = OFAToONNXConverter(quantization_scheme=quantization_scheme)
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing lookup table if available
        self._load_lookup_table()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default hardware configuration."""
        return {
            'execution_providers': ['CPUExecutionProvider'],
            'batch_sizes': [1],
            'input_shapes': [[224, 224]],
            'warmup_runs': 10,
            'measurement_runs': 50,
            'optimization_level': 'all'
        }
    
    def _load_lookup_table(self):
        """Load existing lookup table from cache."""
        cache_file = os.path.join(
            self.cache_dir, 
            f'{self.hardware_name}_{self.quantization_scheme}_latency.yaml'
        )
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.lookup_table = yaml.safe_load(f) or defaultdict(dict)
                logger.info(f"Loaded lookup table with {len(self.lookup_table)} entries")
            except Exception as e:
                logger.warning(f"Failed to load lookup table: {e}")
                self.lookup_table = defaultdict(dict)
    
    def _save_lookup_table(self):
        """Save lookup table to cache."""
        cache_file = os.path.join(
            self.cache_dir,
            f'{self.hardware_name}_{self.quantization_scheme}_latency.yaml'
        )
        try:
            with open(cache_file, 'w') as f:
                yaml.dump(dict(self.lookup_table), f, default_flow_style=False)
            logger.info(f"Saved lookup table to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save lookup table: {e}")
    
    def _measure_latency(
        self, 
        onnx_model_path: str, 
        input_shape: Tuple[int, int, int, int],
        batch_size: int = 1
    ) -> float:
        """
        Measure actual latency of ONNX model on target hardware.
        
        Args:
            onnx_model_path: Path to ONNX model file
            input_shape: Input tensor shape (B, C, H, W)
            batch_size: Batch size for inference
            
        Returns:
            Average latency in milliseconds
        """
        try:
            # Create ONNX Runtime session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            session = ort.InferenceSession(
                onnx_model_path,
                session_options,
                providers=self.providers
            )
            
            # Create test input
            test_input = create_test_input(input_shape, batch_size)
            input_name = session.get_inputs()[0].name
            
            # Warmup runs
            for _ in range(self.config.get('warmup_runs', 10)):
                _ = session.run(None, {input_name: test_input})
            
            # Measurement runs
            latencies = []
            for _ in range(self.config.get('measurement_runs', 50)):
                start_time = time.perf_counter()
                _ = session.run(None, {input_name: test_input})
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            logger.debug(f"Latency: {avg_latency:.3f}±{std_latency:.3f}ms")
            return avg_latency
            
        except Exception as e:
            logger.error(f"Failed to measure latency: {e}")
            return float('inf')
    
    def _get_subnet_key(self, net_config: Dict[str, Any]) -> str:
        """Generate unique key for subnet configuration."""
        key_parts = []
        if 'ks' in net_config:
            key_parts.append(f"ks_{'-'.join(map(str, net_config['ks']))}")
        if 'e' in net_config:
            key_parts.append(f"e_{'-'.join(map(str, net_config['e']))}")
        if 'd' in net_config:
            key_parts.append(f"d_{'-'.join(map(str, net_config['d']))}")
        if 'r' in net_config:
            key_parts.append(f"r_{net_config['r'][0]}")
        return "_".join(key_parts)
    
    def predict_latency(self, net_config: Dict[str, Any]) -> float:
        """
        Predict latency for a given subnet configuration.
        
        Args:
            net_config: Dictionary containing subnet configuration
                       (ks, e, d, r keys for kernel_size, expand_ratio, depth, resolution)
        
        Returns:
            Predicted latency in milliseconds
        """
        subnet_key = self._get_subnet_key(net_config)
        
        # Check if already in lookup table
        if subnet_key in self.lookup_table:
            return self.lookup_table[subnet_key]['latency']
        
        logger.warning(f"Subnet configuration not found in lookup table: {subnet_key}")
        return float('inf')
    
    def build_lookup_table(
        self, 
        ofa_network,
        num_samples: int = 1000,
        batch_size: int = 1,
        save_frequency: int = 100
    ):
        """
        Build latency lookup table by sampling and measuring sub-networks.
        
        Args:
            ofa_network: OFA network instance
            num_samples: Number of sub-network samples to measure
            batch_size: Batch size for latency measurement
            save_frequency: How often to save progress
        """
        logger.info(f"Building latency lookup table with {num_samples} samples")
        
        measured_count = 0
        for i in range(num_samples):
            try:
                # Sample random subnet configuration
                net_config = ofa_network.sample_active_subnet()
                subnet_key = self._get_subnet_key(net_config)
                
                # Skip if already measured
                if subnet_key in self.lookup_table:
                    continue
                
                # Set active subnet
                ofa_network.set_active_subnet(**net_config)
                subnet = ofa_network.get_active_subnet(preserve_weight=True)
                
                # Convert to ONNX and quantize
                onnx_model_path = os.path.join(
                    self.cache_dir, 
                    f'subnet_{i}_{self.quantization_scheme}.onnx'
                )
                
                success = self.onnx_converter.convert_subnet(
                    subnet, 
                    onnx_model_path,
                    input_shape=(batch_size, 3, net_config.get('r', [224])[0], net_config.get('r', [224])[0])
                )
                
                if not success:
                    logger.warning(f"Failed to convert subnet {i} to ONNX")
                    continue
                
                # Measure latency
                input_shape = (batch_size, 3, net_config.get('r', [224])[0], net_config.get('r', [224])[0])
                latency = self._measure_latency(onnx_model_path, input_shape, batch_size)
                
                # Store in lookup table
                self.lookup_table[subnet_key] = {
                    'latency': latency,
                    'config': net_config,
                    'quantization': self.quantization_scheme,
                    'batch_size': batch_size
                }
                
                measured_count += 1
                
                # Clean up temporary ONNX file
                if os.path.exists(onnx_model_path):
                    os.remove(onnx_model_path)
                
                logger.info(f"Sample {i+1}/{num_samples}: {subnet_key} -> {latency:.3f}ms")
                
                # Save progress periodically
                if (i + 1) % save_frequency == 0:
                    self._save_lookup_table()
                    
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        # Final save
        self._save_lookup_table()
        logger.info(f"Completed building lookup table with {measured_count} new measurements")
    
    def get_hardware_efficiency_score(self, net_config: Dict[str, Any]) -> float:
        """
        Get hardware-specific efficiency score for a subnet.
        Higher score means better efficiency for the target hardware.
        
        Args:
            net_config: Subnet configuration
            
        Returns:
            Efficiency score (higher is better)
        """
        latency = self.predict_latency(net_config)
        if latency == float('inf'):
            return 0.0
        
        # Simple efficiency metric: 1 / latency
        # Can be customized based on hardware characteristics
        return 1.0 / latency
    
    def benchmark_suite(
        self, 
        ofa_network,
        configurations: List[Dict[str, Any]],
        batch_sizes: List[int] = [1, 4, 8]
    ) -> Dict[str, Any]:
        """
        Run benchmark suite on predefined configurations.
        
        Args:
            ofa_network: OFA network instance
            configurations: List of subnet configurations to benchmark
            batch_sizes: Batch sizes to test
            
        Returns:
            Benchmark results dictionary
        """
        results = {
            'hardware': self.hardware_name,
            'quantization': self.quantization_scheme,
            'results': []
        }
        
        for config in configurations:
            config_results = {'config': config, 'batch_results': []}
            
            for batch_size in batch_sizes:
                try:
                    # Set active subnet
                    ofa_network.set_active_subnet(**config)
                    subnet = ofa_network.get_active_subnet(preserve_weight=True)
                    
                    # Convert to ONNX
                    onnx_path = os.path.join(
                        self.cache_dir, 
                        f'benchmark_{self.quantization_scheme}_bs{batch_size}.onnx'
                    )
                    
                    input_shape = (batch_size, 3, config.get('r', [224])[0], config.get('r', [224])[0])
                    success = self.onnx_converter.convert_subnet(subnet, onnx_path, input_shape)
                    
                    if success:
                        latency = self._measure_latency(onnx_path, input_shape, batch_size)
                        throughput = batch_size / (latency / 1000.0)  # samples/second
                        
                        config_results['batch_results'].append({
                            'batch_size': batch_size,
                            'latency_ms': latency,
                            'throughput_sps': throughput
                        })
                        
                        # Clean up
                        if os.path.exists(onnx_path):
                            os.remove(onnx_path)
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for config {config}, batch_size {batch_size}: {e}")
            
            results['results'].append(config_results)
        
        return results
