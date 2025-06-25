#!/usr/bin/env python3
"""
End-to-end example for custom hardware integration with OFA

This example demonstrates the complete workflow:
1. Build latency lookup table
2. Train quantized accuracy predictor
3. Search for optimal subnets
4. Evaluate selected subnets

Usage:
    python examples/custom_hardware_example.py --dataset_path /path/to/imagenet --config configs/npu_config.yaml
"""

import argparse
import os
import logging
import json
import time
from pathlib import Path

from ofa.custom_hardware.utils import setup_logging, load_config


def run_command(command: str, description: str = None):
    """Run a shell command and log the output"""
    if description:
        logging.info(f"Running: {description}")
    
    logging.info(f"Command: {command}")
    exit_code = os.system(command)
    
    if exit_code != 0:
        logging.error(f"Command failed with exit code {exit_code}")
        return False
    
    logging.info("Command completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description='End-to-end custom hardware integration example')
    
    # Required arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to ImageNet dataset')
    parser.add_argument('--config', type=str, default='configs/npu_config.yaml',
                       help='Hardware configuration file')
    
    # Model configuration
    parser.add_argument('--ofa_network', type=str, default='ofa_mbv3_d234_e346_k357_w1.0',
                       help='OFA network to use')
    
    # Workflow control
    parser.add_argument('--skip_latency_table', action='store_true',
                       help='Skip building latency lookup table')
    parser.add_argument('--skip_accuracy_predictor', action='store_true',
                       help='Skip training accuracy predictor')
    parser.add_argument('--skip_search', action='store_true',
                       help='Skip evolutionary search')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Skip final evaluation')
    
    # Quick demo mode
    parser.add_argument('--quick_demo', action='store_true',
                       help='Run in quick demo mode (reduced samples/iterations)')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='custom_hardware_results',
                       help='Output directory for all results')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load configuration
    config = load_config(args.config)
    hardware_name = config.get('hardware', {}).get('name', 'CustomHardware')
    
    logger.info("="*60)
    logger.info(f"OFA Custom Hardware Integration Example")
    logger.info(f"Hardware: {hardware_name}")
    logger.info(f"OFA Network: {args.ofa_network}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("="*60)
    
    # Adjust parameters for quick demo
    if args.quick_demo:
        logger.info("Running in quick demo mode...")
        latency_samples = 100
        accuracy_samples = 500
        search_iterations = 100
        population_size = 50
    else:
        latency_samples = 1000
        accuracy_samples = 2000
        search_iterations = 500
        population_size = 100
    
    # Step 1: Build Latency Lookup Table
    if not args.skip_latency_table:
        logger.info("\n" + "="*40)
        logger.info("STEP 1: Building Latency Lookup Table")
        logger.info("="*40)
        
        latency_table_path = output_dir / f"{hardware_name}_latency_table.pkl"
        
        command = f"""python scripts/build_latency_lut.py \\
            --config {args.config} \\
            --model_name {args.ofa_network} \\
            --output_path {latency_table_path} \\
            --num_samples {latency_samples} \\
            --verbose"""
        
        success = run_command(command, "Building latency lookup table")
        if not success:
            logger.error("Failed to build latency lookup table")
            return
        
        logger.info(f"Latency lookup table saved to: {latency_table_path}")
    
    # Step 2: Train Quantized Accuracy Predictor
    if not args.skip_accuracy_predictor:
        logger.info("\n" + "="*40)
        logger.info("STEP 2: Training Quantized Accuracy Predictor")
        logger.info("="*40)
        
        accuracy_predictor_path = output_dir / f"{hardware_name}_accuracy_predictor.pth"
        training_data_path = output_dir / f"{hardware_name}_training_data.json"
        
        command = f"""python scripts/train_accuracy_predictor.py \\
            --dataset_path {args.dataset_path} \\
            --output_path {accuracy_predictor_path} \\
            --config {args.config} \\
            --ofa_network {args.ofa_network} \\
            --num_samples {accuracy_samples} \\
            --save_data {training_data_path} \\
            --quantization \\
            --epochs 200 \\
            --verbose"""
        
        success = run_command(command, "Training quantized accuracy predictor")
        if not success:
            logger.error("Failed to train accuracy predictor")
            return
        
        logger.info(f"Accuracy predictor saved to: {accuracy_predictor_path}")
        logger.info(f"Training data saved to: {training_data_path}")
    
    # Step 3: Search for Optimal Subnets
    if not args.skip_search:
        logger.info("\n" + "="*40)
        logger.info("STEP 3: Searching for Optimal Subnets")
        logger.info("="*40)
        
        latency_table_path = output_dir / f"{hardware_name}_latency_table.pkl"
        accuracy_predictor_path = output_dir / f"{hardware_name}_accuracy_predictor.pth"
        search_results_path = output_dir / f"{hardware_name}_search_results.json"
        
        # Get efficiency constraint from config
        efficiency_constraint = config.get('search', {}).get('efficiency_constraint', 10.0)
        
        command = f"""python scripts/search_custom_hardware.py \\
            --hardware_name {hardware_name} \\
            --latency_constraint {efficiency_constraint} \\
            --ofa_network {args.ofa_network} \\
            --latency_table {latency_table_path} \\
            --accuracy_predictor {accuracy_predictor_path} \\
            --output_path {search_results_path} \\
            --population_size {population_size} \\
            --max_iterations {search_iterations} \\
            --config {args.config} \\
            --verbose"""
        
        success = run_command(command, "Running evolutionary search")
        if not success:
            logger.error("Failed to run evolutionary search")
            return
        
        logger.info(f"Search results saved to: {search_results_path}")
        
        # Extract best subnet for evaluation
        try:
            with open(search_results_path, 'r') as f:
                search_results = json.load(f)
            
            best_subnet = search_results.get('best_architecture')
            if best_subnet:
                best_subnet_path = output_dir / f"{hardware_name}_best_subnet.json"
                with open(best_subnet_path, 'w') as f:
                    json.dump(best_subnet, f, indent=2)
                logger.info(f"Best subnet configuration saved to: {best_subnet_path}")
        except Exception as e:
            logger.warning(f"Could not extract best subnet: {e}")
    
    # Step 4: Evaluate Selected Subnets
    if not args.skip_evaluation:
        logger.info("\n" + "="*40)
        logger.info("STEP 4: Evaluating Selected Subnets")
        logger.info("="*40)
        
        best_subnet_path = output_dir / f"{hardware_name}_best_subnet.json"
        evaluation_results_path = output_dir / f"{hardware_name}_evaluation.json"
        onnx_model_path = output_dir / f"{hardware_name}_best_model.onnx"
        
        if best_subnet_path.exists():
            command = f"""python scripts/evaluate_custom_hardware.py \\
                --subnet_config {best_subnet_path} \\
                --dataset_path {args.dataset_path} \\
                --config {args.config} \\
                --ofa_network {args.ofa_network} \\
                --output_path {evaluation_results_path} \\
                --save_onnx {onnx_model_path} \\
                --quantization \\
                --verbose"""
            
            success = run_command(command, "Evaluating best subnet")
            if not success:
                logger.error("Failed to evaluate subnet")
                return
            
            logger.info(f"Evaluation results saved to: {evaluation_results_path}")
            logger.info(f"ONNX model saved to: {onnx_model_path}")
        else:
            logger.warning(f"Best subnet configuration not found at {best_subnet_path}")
            logger.warning("Skipping evaluation step")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("WORKFLOW COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    
    # Print final results summary
    try:
        evaluation_results_path = output_dir / f"{hardware_name}_evaluation.json"
        if evaluation_results_path.exists():
            with open(evaluation_results_path, 'r') as f:
                results = json.load(f)
            
            pytorch_acc = results.get('pytorch_accuracy', {})
            pytorch_lat = results.get('pytorch_latency', {})
            onnx_eval = results.get('onnx_evaluation', {})
            
            print("\nFINAL RESULTS:")
            print("-" * 40)
            print(f"Hardware: {hardware_name}")
            print(f"OFA Network: {args.ofa_network}")
            
            if pytorch_acc:
                print(f"PyTorch Accuracy: {pytorch_acc.get('top1_accuracy', 0):.2f}%")
            if pytorch_lat:
                print(f"PyTorch Latency: {pytorch_lat.get('mean_latency_ms', 0):.2f} ms")
            
            if onnx_eval and 'error' not in onnx_eval:
                onnx_acc = onnx_eval.get('accuracy', {})
                onnx_lat = onnx_eval.get('latency', {})
                print(f"Quantized ONNX Accuracy: {onnx_acc.get('top1_accuracy', 0):.2f}%")
                print(f"Quantized ONNX Latency: {onnx_lat.get('mean_latency_ms', 0):.2f} ms")
                
                quantization_impact = results.get('quantization_impact', {})
                if quantization_impact:
                    print(f"Quantization Impact: {quantization_impact.get('accuracy_drop_percent', 0):.2f}% accuracy drop")
                    print(f"Performance Speedup: {quantization_impact.get('speedup_ratio', 0):.2f}x")
            
            print("-" * 40)
        
        # Print output file locations
        print("\nGENERATED FILES:")
        print("-" * 40)
        for file_path in output_dir.glob("*"):
            if file_path.is_file():
                print(f"  {file_path}")
        print("-" * 40)
        
    except Exception as e:
        logger.warning(f"Could not load final results: {e}")
    
    logger.info(f"\nAll results are available in: {output_dir}")
    logger.info("You can now use the trained predictors and ONNX models for deployment!")


if __name__ == '__main__':
    main()
