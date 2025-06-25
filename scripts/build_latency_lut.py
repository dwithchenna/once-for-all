#!/usr/bin/env python3
"""
Build latency lookup table for custom NPU hardware

This script builds a comprehensive latency lookup table by sampling
OFA sub-networks, converting them to quantized ONNX models, and 
measuring actual latency on the target hardware.
"""

import argparse
import os
import logging
from typing import List

from ofa.model_zoo import ofa_net
from ofa.custom_hardware import QuantizedLatencyPredictor
from ofa.custom_hardware.utils import setup_logging, create_hardware_config_template


def main():
    parser = argparse.ArgumentParser(description='Build latency lookup table for custom hardware')
    
    # Required arguments
    parser.add_argument('--hardware_name', type=str, required=True,
                       help='Name of the target hardware (e.g., "my_npu")')
    parser.add_argument('--ofa_network', type=str, required=True,
                       choices=['ofa_mbv3_d234_e346_k357_w1.0', 'ofa_mbv3_d234_e346_k357_w1.2',
                               'ofa_proxyless_d234_e346_k357_w1.3', 'ofa_resnet50'],
                       help='OFA network to sample from')
    
    # Optional arguments
    parser.add_argument('--quantization_scheme', type=str, default='int8',
                       choices=['int8', 'fp16', 'dynamic', 'static'],
                       help='Quantization scheme to use')
    parser.add_argument('--num_samples', type=int, default=2000,
                       help='Number of sub-network samples to measure')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for latency measurements')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path for lookup table (default: auto-generated)')
    parser.add_argument('--cache_dir', type=str, default='./latency_cache',
                       help='Directory to cache measurements')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to hardware configuration file')
    parser.add_argument('--providers', type=str, nargs='+', default=None,
                       help='ONNX Runtime execution providers')
    parser.add_argument('--save_frequency', type=int, default=100,
                       help='How often to save progress (number of samples)')
    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Log file path (default: stdout only)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Building latency lookup table for {args.hardware_name}")
    logger.info(f"OFA Network: {args.ofa_network}")
    logger.info(f"Quantization: {args.quantization_scheme}")
    logger.info(f"Samples: {args.num_samples}")
    
    # Create hardware config if it doesn't exist
    if not args.config_path:
        config_dir = './configs'
        os.makedirs(config_dir, exist_ok=True)
        args.config_path = os.path.join(config_dir, f'{args.hardware_name}.yaml')
        
        if not os.path.exists(args.config_path):
            logger.info(f"Creating hardware config template: {args.config_path}")
            create_hardware_config_template(args.hardware_name, args.config_path)
            logger.warning(f"Please review and customize the hardware config: {args.config_path}")
    
    # Load OFA network
    logger.info(f"Loading OFA network: {args.ofa_network}")
    try:
        ofa_network = ofa_net(args.ofa_network, pretrained=True)
        logger.info("OFA network loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load OFA network: {e}")
        return 1
    
    # Setup latency predictor
    providers = args.providers
    if providers is None:
        # Default providers based on hardware name
        if 'npu' in args.hardware_name.lower():
            providers = [f'{args.hardware_name}ExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
    
    try:
        latency_predictor = QuantizedLatencyPredictor(
            hardware_name=args.hardware_name,
            quantization_scheme=args.quantization_scheme,
            onnx_runtime_providers=providers,
            config_path=args.config_path,
            cache_dir=args.cache_dir
        )
        logger.info("Latency predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize latency predictor: {e}")
        return 1
    
    # Build lookup table
    try:
        logger.info("Starting lookup table construction...")
        latency_predictor.build_lookup_table(
            ofa_network=ofa_network,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            save_frequency=args.save_frequency
        )
        logger.info("Lookup table construction completed successfully")
    except Exception as e:
        logger.error(f"Failed to build lookup table: {e}")
        return 1
    
    # Save final results
    if args.output_path:
        output_path = args.output_path
    else:
        output_dir = './latency_tables'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f'{args.hardware_name}_{args.quantization_scheme}_{args.ofa_network}.yaml'
        )
    
    try:
        # Copy from cache to final location
        import shutil
        cache_file = os.path.join(
            args.cache_dir,
            f'{args.hardware_name}_{args.quantization_scheme}_latency.yaml'
        )
        
        if os.path.exists(cache_file):
            shutil.copy2(cache_file, output_path)
            logger.info(f"Lookup table saved to: {output_path}")
        else:
            logger.warning("Cache file not found, lookup table may not have been saved")
    except Exception as e:
        logger.error(f"Failed to save final lookup table: {e}")
        return 1
    
    logger.info("Latency lookup table build completed successfully!")
    return 0


if __name__ == '__main__':
    exit(main())
