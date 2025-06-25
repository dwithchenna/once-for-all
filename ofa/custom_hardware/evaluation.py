"""
Evaluation module for quantized models on custom hardware
"""

import os
import time
import torch
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm

try:
    from torchvision import transforms, datasets
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from .utils import create_test_input, save_results
from .onnx_converter import OFAToONNXConverter

logger = logging.getLogger(__name__)


def evaluate_quantized_model(
    onnx_model_path: str,
    dataset_path: Optional[str] = None,
    providers: List[str] = ['CPUExecutionProvider'],
    batch_size: int = 1,
    num_samples: Optional[int] = None,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)
) -> Dict[str, float]:
    """
    Evaluate quantized ONNX model accuracy and performance.
    
    Args:
        onnx_model_path: Path to ONNX model
        dataset_path: Path to evaluation dataset (ImageNet format)
        providers: ONNX Runtime execution providers
        batch_size: Batch size for evaluation
        num_samples: Number of samples to evaluate (None for full dataset)
        input_shape: Input tensor shape
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Create ONNX Runtime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            onnx_model_path,
            session_options,
            providers=providers
        )
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        if dataset_path and HAS_TORCHVISION:
            # Evaluate on real dataset
            metrics = _evaluate_on_dataset(
                session, input_name, output_name, 
                dataset_path, input_shape, batch_size, num_samples
            )
        else:
            # Evaluate on synthetic data
            logger.warning("Evaluating on synthetic data - results may not be meaningful")
            metrics = _evaluate_on_synthetic_data(
                session, input_name, output_name,
                input_shape, batch_size, num_samples or 100
            )
        
        # Add model info
        metrics['model_path'] = onnx_model_path
        metrics['providers'] = providers
        metrics['batch_size'] = batch_size
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        return {'error': str(e)}


def _evaluate_on_dataset(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    dataset_path: str,
    input_shape: Tuple[int, int, int, int],
    batch_size: int,
    num_samples: Optional[int]
) -> Dict[str, float]:
    """Evaluate model on real dataset."""
    
    if not HAS_TORCHVISION:
        raise ImportError("torchvision required for dataset evaluation")
    
    # Setup data transforms
    image_size = input_shape[2]
    transform = transforms.Compose([
        transforms.Resize(int(image_size / 0.875)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'val'),
        transform=transform
    )
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluation metrics
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    total_time = 0.0
    
    samples_processed = 0
    max_samples = num_samples or len(dataset)
    
    with tqdm(total=min(max_samples, len(dataset)), desc="Evaluating") as pbar:
        for batch_idx, (images, labels) in enumerate(data_loader):
            if samples_processed >= max_samples:
                break
            
            # Convert to numpy
            images_np = images.numpy()
            labels_np = labels.numpy()
            
            # Measure inference time
            start_time = time.perf_counter()
            outputs = session.run([output_name], {input_name: images_np})[0]
            end_time = time.perf_counter()
            
            total_time += (end_time - start_time)
            
            # Calculate accuracy
            batch_size_actual = images_np.shape[0]
            top1_correct, top5_correct = _calculate_accuracy(outputs, labels_np)
            
            correct_top1 += top1_correct
            correct_top5 += top5_correct
            total_samples += batch_size_actual
            samples_processed += batch_size_actual
            
            # Update progress
            pbar.update(batch_size_actual)
            pbar.set_postfix({
                'Top1': f"{100.0 * correct_top1 / total_samples:.2f}%",
                'Top5': f"{100.0 * correct_top5 / total_samples:.2f}%"
            })
    
    # Calculate final metrics
    top1_accuracy = 100.0 * correct_top1 / total_samples
    top5_accuracy = 100.0 * correct_top5 / total_samples
    avg_latency_ms = (total_time / total_samples) * 1000
    throughput_sps = total_samples / total_time
    
    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'avg_latency_ms': avg_latency_ms,
        'throughput_sps': throughput_sps,
        'total_samples': total_samples,
        'evaluation_time_s': total_time
    }


def _evaluate_on_synthetic_data(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    input_shape: Tuple[int, int, int, int],
    batch_size: int,
    num_samples: int
) -> Dict[str, float]:
    """Evaluate model on synthetic data (for performance testing only)."""
    
    total_time = 0.0
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for _ in range(num_batches):
        # Create synthetic input
        test_input = create_test_input(input_shape, batch_size)
        
        # Measure inference time
        start_time = time.perf_counter()
        outputs = session.run([output_name], {input_name: test_input})[0]
        end_time = time.perf_counter()
        
        total_time += (end_time - start_time)
    
    avg_latency_ms = (total_time / num_samples) * 1000
    throughput_sps = num_samples / total_time
    
    return {
        'avg_latency_ms': avg_latency_ms,
        'throughput_sps': throughput_sps,
        'total_samples': num_samples,
        'evaluation_time_s': total_time,
        'note': 'Evaluated on synthetic data - accuracy metrics not available'
    }


def _calculate_accuracy(outputs: np.ndarray, labels: np.ndarray) -> Tuple[int, int]:
    """Calculate top-1 and top-5 accuracy."""
    batch_size = outputs.shape[0]
    
    # Get top-5 predictions
    top5_pred = np.argsort(outputs, axis=1)[:, -5:]
    
    top1_correct = 0
    top5_correct = 0
    
    for i in range(batch_size):
        true_label = labels[i]
        
        # Top-1 accuracy
        if top5_pred[i, -1] == true_label:
            top1_correct += 1
        
        # Top-5 accuracy
        if true_label in top5_pred[i]:
            top5_correct += 1
    
    return top1_correct, top5_correct


def benchmark_subnet_on_hardware(
    ofa_network,
    subnet_config: Dict[str, Any],
    hardware_name: str,
    quantization_scheme: str = 'int8',
    dataset_path: Optional[str] = None,
    providers: List[str] = ['CPUExecutionProvider'],
    batch_sizes: List[int] = [1, 4, 8],
    num_samples: int = 1000,
    output_dir: str = './benchmark_results'
) -> Dict[str, Any]:
    """
    Comprehensive benchmark of a subnet on custom hardware.
    
    Args:
        ofa_network: OFA network instance
        subnet_config: Subnet configuration
        hardware_name: Name of target hardware
        quantization_scheme: Quantization scheme
        dataset_path: Path to evaluation dataset
        providers: ONNX Runtime execution providers
        batch_sizes: List of batch sizes to test
        num_samples: Number of samples for evaluation
        output_dir: Directory to save results
        
    Returns:
        Comprehensive benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    benchmark_results = {
        'subnet_config': subnet_config,
        'hardware_name': hardware_name,
        'quantization_scheme': quantization_scheme,
        'batch_results': [],
        'summary': {}
    }
    
    # Convert subnet to ONNX
    converter = OFAToONNXConverter(quantization_scheme=quantization_scheme)
    
    # Set active subnet
    ofa_network.set_active_subnet(**subnet_config)
    subnet = ofa_network.get_active_subnet(preserve_weight=True)
    
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch size {batch_size}")
        
        try:
            # Convert to ONNX for this batch size
            input_shape = (batch_size, 3, subnet_config.get('r', [224])[0], subnet_config.get('r', [224])[0])
            onnx_path = os.path.join(
                output_dir, 
                f'{hardware_name}_{quantization_scheme}_bs{batch_size}.onnx'
            )
            
            success = converter.convert_subnet(subnet, onnx_path, input_shape)
            
            if not success:
                logger.error(f"Failed to convert subnet for batch size {batch_size}")
                continue
            
            # Evaluate
            metrics = evaluate_quantized_model(
                onnx_path,
                dataset_path=dataset_path,
                providers=providers,
                batch_size=batch_size,
                num_samples=num_samples,
                input_shape=input_shape
            )
            
            # Add batch size info
            metrics['batch_size'] = batch_size
            benchmark_results['batch_results'].append(metrics)
            
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
                
        except Exception as e:
            logger.error(f"Benchmark failed for batch size {batch_size}: {e}")
    
    # Calculate summary metrics
    if benchmark_results['batch_results']:
        # Find best batch size for throughput
        best_throughput = max(
            benchmark_results['batch_results'], 
            key=lambda x: x.get('throughput_sps', 0)
        )
        
        # Find best batch size for latency
        best_latency = min(
            benchmark_results['batch_results'],
            key=lambda x: x.get('avg_latency_ms', float('inf'))
        )
        
        benchmark_results['summary'] = {
            'best_throughput': {
                'batch_size': best_throughput['batch_size'],
                'throughput_sps': best_throughput.get('throughput_sps', 0),
                'latency_ms': best_throughput.get('avg_latency_ms', 0)
            },
            'best_latency': {
                'batch_size': best_latency['batch_size'],
                'latency_ms': best_latency.get('avg_latency_ms', float('inf')),
                'throughput_sps': best_latency.get('throughput_sps', 0)
            },
            'accuracy_top1': benchmark_results['batch_results'][0].get('top1_accuracy'),
            'accuracy_top5': benchmark_results['batch_results'][0].get('top5_accuracy')
        }
    
    # Save results
    results_path = os.path.join(
        output_dir,
        f'benchmark_{hardware_name}_{quantization_scheme}.json'
    )
    save_results(benchmark_results, results_path)
    
    return benchmark_results


def compare_quantization_schemes(
    ofa_network,
    subnet_config: Dict[str, Any],
    hardware_name: str,
    quantization_schemes: List[str] = ['fp32', 'fp16', 'int8'],
    dataset_path: Optional[str] = None,
    providers: List[str] = ['CPUExecutionProvider'],
    batch_size: int = 1,
    num_samples: int = 1000,
    output_dir: str = './quantization_comparison'
) -> Dict[str, Any]:
    """
    Compare different quantization schemes for a subnet.
    
    Args:
        ofa_network: OFA network instance
        subnet_config: Subnet configuration
        hardware_name: Name of target hardware
        quantization_schemes: List of quantization schemes to compare
        dataset_path: Path to evaluation dataset
        providers: ONNX Runtime execution providers
        batch_size: Batch size for evaluation
        num_samples: Number of samples for evaluation
        output_dir: Directory to save results
        
    Returns:
        Comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_results = {
        'subnet_config': subnet_config,
        'hardware_name': hardware_name,
        'quantization_results': {},
        'summary': {}
    }
    
    # Set active subnet
    ofa_network.set_active_subnet(**subnet_config)
    subnet = ofa_network.get_active_subnet(preserve_weight=True)
    
    input_shape = (batch_size, 3, subnet_config.get('r', [224])[0], subnet_config.get('r', [224])[0])
    
    for quant_scheme in quantization_schemes:
        logger.info(f"Evaluating quantization scheme: {quant_scheme}")
        
        try:
            # Convert to ONNX with specific quantization
            converter = OFAToONNXConverter(quantization_scheme=quant_scheme)
            onnx_path = os.path.join(
                output_dir,
                f'{hardware_name}_{quant_scheme}.onnx'
            )
            
            success = converter.convert_subnet(subnet, onnx_path, input_shape)
            
            if not success:
                logger.error(f"Failed to convert subnet with {quant_scheme} quantization")
                continue
            
            # Evaluate
            metrics = evaluate_quantized_model(
                onnx_path,
                dataset_path=dataset_path,
                providers=providers,
                batch_size=batch_size,
                num_samples=num_samples,
                input_shape=input_shape
            )
            
            # Get model size
            model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            metrics['model_size_mb'] = model_size_mb
            
            comparison_results['quantization_results'][quant_scheme] = metrics
            
            # Clean up
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
                
        except Exception as e:
            logger.error(f"Comparison failed for {quant_scheme}: {e}")
    
    # Calculate summary
    if comparison_results['quantization_results']:
        # Find best accuracy
        best_accuracy = max(
            comparison_results['quantization_results'].items(),
            key=lambda x: x[1].get('top1_accuracy', 0)
        )
        
        # Find best performance
        best_performance = max(
            comparison_results['quantization_results'].items(),
            key=lambda x: x[1].get('throughput_sps', 0)
        )
        
        # Find smallest model
        smallest_model = min(
            comparison_results['quantization_results'].items(),
            key=lambda x: x[1].get('model_size_mb', float('inf'))
        )
        
        comparison_results['summary'] = {
            'best_accuracy': {
                'scheme': best_accuracy[0],
                'top1_accuracy': best_accuracy[1].get('top1_accuracy', 0)
            },
            'best_performance': {
                'scheme': best_performance[0],
                'throughput_sps': best_performance[1].get('throughput_sps', 0)
            },
            'smallest_model': {
                'scheme': smallest_model[0],
                'model_size_mb': smallest_model[1].get('model_size_mb', 0)
            }
        }
    
    # Save results
    results_path = os.path.join(
        output_dir,
        f'quantization_comparison_{hardware_name}.json'
    )
    save_results(comparison_results, results_path)
    
    return comparison_results
