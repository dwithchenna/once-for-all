"""
OFA to ONNX Converter with Quantization Support

This module converts OFA sub-networks to ONNX format with quantization
for deployment on custom NPU hardware.
"""

import os
import torch
import onnx
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

try:
    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader
    from onnxruntime.quantization.shape_inference import quant_pre_process
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False
    logging.warning("ONNX Runtime quantization not available. Install onnxruntime with quantization support.")

logger = logging.getLogger(__name__)


class CalibrationDataset(CalibrationDataReader):
    """Calibration dataset for static quantization."""
    
    def __init__(self, dataset_path: str, input_shape: Tuple[int, int, int, int], num_samples: int = 100):
        super().__init__()
        self.dataset_path = dataset_path
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.data_iterator = self._create_data_iterator()
    
    def _create_data_iterator(self):
        """Create iterator over calibration data."""
        # For simplicity, using random data. In practice, use real dataset
        for i in range(self.num_samples):
            # Generate random input tensor
            data = np.random.rand(*self.input_shape).astype(np.float32)
            yield {'input': data}
    
    def get_next(self):
        """Get next calibration sample."""
        try:
            return next(self.data_iterator)
        except StopIteration:
            return None


class OFAToONNXConverter:
    """
    Converts OFA sub-networks to quantized ONNX models.
    
    Supports multiple quantization schemes and optimization levels
    for different hardware targets.
    """
    
    def __init__(
        self, 
        quantization_scheme: str = 'int8',
        quantization_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ONNX converter.
        
        Args:
            quantization_scheme: Quantization scheme ('int8', 'fp16', 'dynamic', 'static')
            quantization_config: Custom quantization configuration
        """
        self.quantization_scheme = quantization_scheme
        self.quantization_config = quantization_config or self._get_default_quantization_config()
        
        if not HAS_QUANTIZATION and quantization_scheme in ['int8', 'static', 'dynamic']:
            raise ImportError("ONNX Runtime quantization not available. Please install onnxruntime with quantization support.")
    
    def _get_default_quantization_config(self) -> Dict[str, Any]:
        """Get default quantization configuration."""
        configs = {
            'int8': {
                'activation_type': QuantType.QInt8 if HAS_QUANTIZATION else 'QInt8',
                'weight_type': QuantType.QInt8 if HAS_QUANTIZATION else 'QInt8',
                'per_channel': True,
                'reduce_range': False,
                'optimization_level': 99
            },
            'fp16': {
                'keep_io_types': True,
            },
            'dynamic': {
                'weight_type': QuantType.QInt8 if HAS_QUANTIZATION else 'QInt8',
                'per_channel': True,
                'reduce_range': False,
                'optimization_level': 99
            },
            'static': {
                'activation_type': QuantType.QInt8 if HAS_QUANTIZATION else 'QInt8',
                'weight_type': QuantType.QInt8 if HAS_QUANTIZATION else 'QInt8',
                'per_channel': True,
                'reduce_range': False,
                'optimization_level': 99,
                'calibration_method': 'entropy'
            }
        }
        return configs.get(self.quantization_scheme, configs['int8'])
    
    def convert_subnet(
        self, 
        subnet_model: torch.nn.Module,
        output_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        opset_version: int = 11,
        calibration_dataset_path: Optional[str] = None
    ) -> bool:
        """
        Convert OFA subnet to quantized ONNX model.
        
        Args:
            subnet_model: PyTorch subnet model
            output_path: Path to save ONNX model
            input_shape: Input tensor shape (B, C, H, W)
            opset_version: ONNX opset version
            calibration_dataset_path: Path to calibration dataset (for static quantization)
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Ensure model is in eval mode
            subnet_model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            
            # Create temporary unquantized ONNX file
            temp_onnx_path = output_path.replace('.onnx', '_temp.onnx')
            
            # Export to ONNX
            torch.onnx.export(
                subnet_model,
                dummy_input,
                temp_onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Apply quantization
            if self.quantization_scheme == 'none':
                # No quantization, just copy the file
                os.rename(temp_onnx_path, output_path)
            elif self.quantization_scheme == 'fp16':
                self._apply_fp16_quantization(temp_onnx_path, output_path)
            elif self.quantization_scheme == 'dynamic':
                self._apply_dynamic_quantization(temp_onnx_path, output_path)
            elif self.quantization_scheme in ['int8', 'static']:
                self._apply_static_quantization(
                    temp_onnx_path, 
                    output_path, 
                    input_shape,
                    calibration_dataset_path
                )
            else:
                logger.error(f"Unsupported quantization scheme: {self.quantization_scheme}")
                return False
            
            # Clean up temporary file
            if os.path.exists(temp_onnx_path):
                os.remove(temp_onnx_path)
            
            # Verify the output model
            if self._verify_onnx_model(output_path):
                logger.info(f"Successfully converted subnet to {output_path}")
                return True
            else:
                logger.error(f"ONNX model verification failed: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to convert subnet to ONNX: {e}")
            return False
    
    def _apply_fp16_quantization(self, input_path: str, output_path: str):
        """Apply FP16 quantization."""
        from onnxruntime.transformers.float16 import convert_float_to_float16
        
        # Load model
        model = onnx.load(input_path)
        
        # Convert to FP16
        model_fp16 = convert_float_to_float16(
            model, 
            keep_io_types=self.quantization_config.get('keep_io_types', True)
        )
        
        # Save
        onnx.save(model_fp16, output_path)
    
    def _apply_dynamic_quantization(self, input_path: str, output_path: str):
        """Apply dynamic quantization."""
        if not HAS_QUANTIZATION:
            raise ImportError("ONNX Runtime quantization not available")
        
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=self.quantization_config['weight_type'],
            per_channel=self.quantization_config.get('per_channel', True),
            reduce_range=self.quantization_config.get('reduce_range', False),
            optimization_level=self.quantization_config.get('optimization_level', 99)
        )
    
    def _apply_static_quantization(
        self, 
        input_path: str, 
        output_path: str,
        input_shape: Tuple[int, int, int, int],
        calibration_dataset_path: Optional[str] = None
    ):
        """Apply static quantization."""
        if not HAS_QUANTIZATION:
            raise ImportError("ONNX Runtime quantization not available")
        
        # Preprocess for quantization
        preprocessed_path = input_path.replace('.onnx', '_preprocessed.onnx')
        quant_pre_process(input_path, preprocessed_path)
        
        # Create calibration data reader
        if calibration_dataset_path:
            # Use real dataset for calibration
            calibration_data_reader = CalibrationDataset(
                calibration_dataset_path, 
                input_shape, 
                num_samples=self.quantization_config.get('calibration_samples', 100)
            )
        else:
            # Use random data for calibration (not recommended for production)
            calibration_data_reader = CalibrationDataset(
                None, 
                input_shape, 
                num_samples=100
            )
        
        # Apply static quantization
        quantize_static(
            model_input=preprocessed_path,
            model_output=output_path,
            calibration_data_reader=calibration_data_reader,
            quant_format=self.quantization_config.get('quant_format', 'QOperator'),
            activation_type=self.quantization_config['activation_type'],
            weight_type=self.quantization_config['weight_type'],
            per_channel=self.quantization_config.get('per_channel', True),
            reduce_range=self.quantization_config.get('reduce_range', False),
            optimization_level=self.quantization_config.get('optimization_level', 99)
        )
        
        # Clean up
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
    
    def _verify_onnx_model(self, model_path: str) -> bool:
        """Verify ONNX model is valid."""
        try:
            # Load and check model
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            
            # Check if model can be loaded in ONNX Runtime
            import onnxruntime as ort
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            
            return True
        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            return False
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get information about ONNX model."""
        try:
            model = onnx.load(model_path)
            
            # Get input/output info
            inputs = [(inp.name, inp.type.tensor_type.shape) for inp in model.graph.input]
            outputs = [(out.name, out.type.tensor_type.shape) for out in model.graph.output]
            
            # Count parameters
            param_count = sum(
                np.prod([dim.dim_value for dim in init.dims]) 
                for init in model.graph.initializer
            )
            
            # Get model size
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            return {
                'inputs': inputs,
                'outputs': outputs,
                'parameter_count': int(param_count),
                'model_size_mb': model_size_mb,
                'quantization_scheme': self.quantization_scheme
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
    
    def batch_convert(
        self,
        ofa_network,
        subnet_configs: List[Dict[str, Any]],
        output_dir: str,
        batch_size: int = 1
    ) -> List[str]:
        """
        Convert multiple subnet configurations to ONNX models.
        
        Args:
            ofa_network: OFA network instance
            subnet_configs: List of subnet configurations
            output_dir: Directory to save ONNX models
            batch_size: Batch size for converted models
            
        Returns:
            List of successfully converted model paths
        """
        os.makedirs(output_dir, exist_ok=True)
        converted_models = []
        
        for i, config in enumerate(subnet_configs):
            try:
                # Set active subnet
                ofa_network.set_active_subnet(**config)
                subnet = ofa_network.get_active_subnet(preserve_weight=True)
                
                # Generate output path
                config_str = "_".join([
                    f"ks{'-'.join(map(str, config.get('ks', [])))}",
                    f"e{'-'.join(map(str, config.get('e', [])))}",
                    f"d{'-'.join(map(str, config.get('d', [])))}",
                    f"r{config.get('r', [224])[0]}"
                ])
                output_path = os.path.join(
                    output_dir, 
                    f"subnet_{i}_{config_str}_{self.quantization_scheme}.onnx"
                )
                
                # Convert to ONNX
                input_shape = (batch_size, 3, config.get('r', [224])[0], config.get('r', [224])[0])
                success = self.convert_subnet(subnet, output_path, input_shape)
                
                if success:
                    converted_models.append(output_path)
                    logger.info(f"Converted {i+1}/{len(subnet_configs)}: {output_path}")
                else:
                    logger.warning(f"Failed to convert subnet {i+1}/{len(subnet_configs)}")
                    
            except Exception as e:
                logger.error(f"Error converting subnet {i}: {e}")
        
        logger.info(f"Successfully converted {len(converted_models)}/{len(subnet_configs)} models")
        return converted_models
