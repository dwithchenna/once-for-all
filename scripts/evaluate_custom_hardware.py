#!/usr/bin/env python3
"""
Evaluate OFA sub-networks on custom hardware

This script evaluates selected OFA sub-networks on custom NPU hardware,
measuring both accuracy and latency with quantization.
"""

import argparse
import os
import logging
import json
import time
from typing import Dict, Any, List
import numpy as np

import torch
import torch.utils.data
from torchvision import transforms, datasets

from ofa.model_zoo import ofa_net
from ofa.custom_hardware import (
    ONNXConverter,
    QuantizedLatencyPredictor, 
    evaluate_quantized_model
)
from ofa.custom_hardware.utils import setup_logging, load_config, generate_test_input
from ofa.utils import AverageMeter, accuracy


def create_data_loader(dataset_path: str, batch_size: int = 50, num_workers: int = 4, 
                      image_size: int = 224) -> torch.utils.data.DataLoader:
    """Create ImageNet validation data loader"""
    transform = transforms.Compose([
        transforms.Resize(int(image_size / 0.875)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'val'),
        transform=transform
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


def evaluate_pytorch_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                          device: str = 'cuda') -> Dict[str, float]:
    """Evaluate PyTorch model accuracy"""
    model.eval()
    model = model.to(device)
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Measure accuracy
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
    
    return {
        'top1_accuracy': top1.avg,
        'top5_accuracy': top5.avg
    }


def measure_pytorch_latency(model: torch.nn.Module, input_shape: tuple,
                           device: str = 'cuda', num_runs: int = 100,
                           warmup_runs: int = 10) -> Dict[str, float]:
    """Measure PyTorch model latency"""
    model.eval()
    model = model.to(device)
    
    # Generate test input
    test_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(test_input)
    
    # Measure latency
    torch.cuda.synchronize() if device == 'cuda' else None
    
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(test_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate OFA sub-networks on custom hardware')
    
    # Required arguments
    parser.add_argument('--subnet_config', type=str, required=True,
                       help='Path to subnet configuration JSON file')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to ImageNet dataset')
    
    # Model configuration
    parser.add_argument('--ofa_network', type=str, default='ofa_mbv3_d234_e346_k357_w1.0',
                       help='OFA network name')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Hardware configuration  
    parser.add_argument('--config', type=str, default=None,
                       help='Hardware configuration YAML file')
    parser.add_argument('--provider', type=str, default='CPUExecutionProvider',
                       help='ONNX Runtime execution provider')
    parser.add_argument('--quantization', action='store_true',
                       help='Enable quantization')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--latency_runs', type=int, default=100,
                       help='Number of runs for latency measurement')
    parser.add_argument('--warmup_runs', type=int, default=10,
                       help='Number of warmup runs')
    
    # Output
    parser.add_argument('--output_path', type=str, default='evaluation_results.json',
                       help='Path to save evaluation results')
    parser.add_argument('--save_onnx', type=str, default=None,
                       help='Path to save ONNX model')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO if args.verbose else logging.WARNING)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config) if args.config else {}
    
    # Override config with command line arguments
    config.setdefault('hardware', {})
    config['hardware']['provider'] = args.provider
    config.setdefault('quantization', {})
    config['quantization']['enabled'] = args.quantization
    
    logger.info(f"Loading subnet configuration from {args.subnet_config}")
    with open(args.subnet_config, 'r') as f:
        subnet_config = json.load(f)
    
    # Load OFA network
    logger.info(f"Loading OFA network: {args.ofa_network}")
    ofa_network = ofa_net(args.ofa_network, pretrained=True)
    
    # Set subnet configuration
    if hasattr(ofa_network, 'set_active_subnet'):
        ofa_network.set_active_subnet(**subnet_config)
    
    # Get the active subnet
    subnet = ofa_network.get_active_subnet() if hasattr(ofa_network, 'get_active_subnet') else ofa_network
    
    # Create data loader
    logger.info(f"Creating data loader for dataset: {args.dataset_path}")
    data_loader = create_data_loader(
        args.dataset_path, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    results = {
        'subnet_config': subnet_config,
        'ofa_network': args.ofa_network,
        'image_size': args.image_size,
        'quantization_enabled': args.quantization
    }
    
    # Evaluate PyTorch model
    logger.info("Evaluating PyTorch model accuracy...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pytorch_accuracy = evaluate_pytorch_model(subnet, data_loader, device)
    results['pytorch_accuracy'] = pytorch_accuracy
    
    logger.info("Measuring PyTorch model latency...")
    input_shape = (1, 3, args.image_size, args.image_size)
    pytorch_latency = measure_pytorch_latency(
        subnet, input_shape, device, 
        args.latency_runs, args.warmup_runs
    )
    results['pytorch_latency'] = pytorch_latency
    
    # Convert to ONNX and evaluate
    logger.info("Converting to ONNX...")
    converter = ONNXConverter(
        quantization_enabled=args.quantization,
        config=config.get('quantization', {})
    )
    
    try:
        onnx_model_path = converter.convert_subnet(
            subnet,
            subnet_config,
            input_shape,
            output_path=args.save_onnx
        )
        
        logger.info("Evaluating ONNX model...")
        onnx_results = evaluate_quantized_model(
            onnx_model_path,
            data_loader,
            provider=args.provider,
            provider_options=config.get('hardware', {}).get('provider_options', {}),
            image_size=args.image_size,
            num_latency_runs=args.latency_runs,
            warmup_runs=args.warmup_runs
        )
        
        results['onnx_evaluation'] = onnx_results
        
        # Calculate quantization impact
        if args.quantization:
            accuracy_drop = pytorch_accuracy['top1_accuracy'] - onnx_results['accuracy']['top1_accuracy']
            speedup = pytorch_latency['mean_latency_ms'] / onnx_results['latency']['mean_latency_ms']
            
            results['quantization_impact'] = {
                'accuracy_drop_percent': accuracy_drop,
                'speedup_ratio': speedup
            }
            
            logger.info(f"Quantization impact - Accuracy drop: {accuracy_drop:.2f}%, Speedup: {speedup:.2f}x")
        
    except Exception as e:
        logger.error(f"ONNX evaluation failed: {str(e)}")
        results['onnx_evaluation'] = {'error': str(e)}
    
    # Save results
    logger.info(f"Saving results to {args.output_path}")
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Subnet configuration: {subnet_config}")
    print(f"PyTorch accuracy: Top-1: {pytorch_accuracy['top1_accuracy']:.2f}%, Top-5: {pytorch_accuracy['top5_accuracy']:.2f}%")
    print(f"PyTorch latency: {pytorch_latency['mean_latency_ms']:.2f} ± {pytorch_latency['std_latency_ms']:.2f} ms")
    
    if 'onnx_evaluation' in results and 'error' not in results['onnx_evaluation']:
        onnx_acc = results['onnx_evaluation']['accuracy']
        onnx_lat = results['onnx_evaluation']['latency']
        print(f"ONNX accuracy: Top-1: {onnx_acc['top1_accuracy']:.2f}%, Top-5: {onnx_acc['top5_accuracy']:.2f}%")
        print(f"ONNX latency: {onnx_lat['mean_latency_ms']:.2f} ± {onnx_lat['std_latency_ms']:.2f} ms")
        
        if args.quantization and 'quantization_impact' in results:
            impact = results['quantization_impact']
            print(f"Quantization impact: {impact['accuracy_drop_percent']:.2f}% accuracy drop, {impact['speedup_ratio']:.2f}x speedup")
    
    print("="*50)


if __name__ == '__main__':
    main()
