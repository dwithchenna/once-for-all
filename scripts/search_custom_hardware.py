#!/usr/bin/env python3
"""
Search for optimal sub-networks on custom hardware

This script uses evolutionary search to find optimal OFA sub-networks
for deployment on custom NPU hardware with quantization constraints.
"""

import argparse
import os
import logging
import json
from typing import List, Dict, Any

from ofa.model_zoo import ofa_net
from ofa.custom_hardware import (
    QuantizedLatencyPredictor, 
    QuantizedAccuracyPredictor,
    CustomHardwareEvolutionFinder
)
from ofa.custom_hardware.utils import setup_logging, save_results


def main():
    parser = argparse.ArgumentParser(description='Search optimal sub-networks for custom hardware')
    
    # Required arguments
    parser.add_argument('--hardware_name', type=str, required=True,
                       help='Name of the target hardware')
    parser.add_argument('--latency_constraint', type=float, required=True,
                       help='Maximum allowed latency in milliseconds')
    
    # Network and quantization
    parser.add_argument('--ofa_network', type=str, default='ofa_mbv3_d234_e346_k357_w1.0',
                       choices=['ofa_mbv3_d234_e346_k357_w1.0', 'ofa_mbv3_d234_e346_k357_w1.2',
                               'ofa_proxyless_d234_e346_k357_w1.3', 'ofa_resnet50'],
                       help='OFA network to search')
    parser.add_argument('--quantization_scheme', type=str, default='int8',
                       choices=['int8', 'fp16', 'dynamic', 'static'],
                       help='Quantization scheme')
    
    # Search parameters
    parser.add_argument('--population_size', type=int, default=200,
                       help='Population size for evolutionary search')
    parser.add_argument('--max_iterations', type=int, default=1000,
                       help='Maximum number of search iterations')
    parser.add_argument('--mutation_prob', type=float, default=0.1,
                       help='Mutation probability')
    parser.add_argument('--crossover_prob', type=float, default=0.8,
                       help='Crossover probability')
    parser.add_argument('--elite_ratio', type=float, default=0.2,
                       help='Ratio of elite individuals to preserve')
    
    # Objectives and constraints
    parser.add_argument('--objectives', type=str, nargs='+', 
                       default=['accuracy'],
                       choices=['accuracy', 'latency', 'efficiency', 'hardware_efficiency'],
                       help='Optimization objectives')
    parser.add_argument('--accuracy_threshold', type=float, default=None,
                       help='Minimum required accuracy percentage')
    parser.add_argument('--power_constraint', type=float, default=None,
                       help='Maximum power consumption in mW')
    
    # Paths and configuration
    parser.add_argument('--latency_table_path', type=str, default=None,
                       help='Path to pre-built latency lookup table')
    parser.add_argument('--accuracy_predictor_path', type=str, default=None,
                       help='Path to custom accuracy predictor model')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to hardware configuration file')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path for search results')
    parser.add_argument('--providers', type=str, nargs='+', default=None,
                       help='ONNX Runtime execution providers')
    
    # Progress and logging
    parser.add_argument('--save_progress', action='store_true',
                       help='Save progress during search')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output during search')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Searching optimal sub-networks for {args.hardware_name}")
    logger.info(f"Latency constraint: {args.latency_constraint}ms")
    logger.info(f"Objectives: {args.objectives}")
    
    # Load OFA network
    logger.info(f"Loading OFA network: {args.ofa_network}")
    try:
        ofa_network = ofa_net(args.ofa_network, pretrained=True)
        logger.info("OFA network loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load OFA network: {e}")
        return 1
    
    # Setup providers
    providers = args.providers
    if providers is None:
        if 'npu' in args.hardware_name.lower():
            providers = [f'{args.hardware_name}ExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
    
    # Initialize latency predictor
    logger.info("Initializing latency predictor...")
    try:
        latency_predictor = QuantizedLatencyPredictor(
            hardware_name=args.hardware_name,
            quantization_scheme=args.quantization_scheme,
            onnx_runtime_providers=providers,
            config_path=args.config_path
        )
        
        # Load pre-built lookup table if available
        if args.latency_table_path and os.path.exists(args.latency_table_path):
            import yaml
            with open(args.latency_table_path, 'r') as f:
                latency_predictor.lookup_table = yaml.safe_load(f)
            logger.info(f"Loaded latency lookup table: {args.latency_table_path}")
        else:
            logger.warning("No latency lookup table provided. Build one with build_latency_lut.py first.")
            
        logger.info("Latency predictor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize latency predictor: {e}")
        return 1
    
    # Initialize accuracy predictor
    logger.info("Initializing accuracy predictor...")
    try:
        accuracy_predictor = QuantizedAccuracyPredictor(
            quantization_scheme=args.quantization_scheme,
            hardware_name=args.hardware_name,
            pretrained=True,
            model_path=args.accuracy_predictor_path
        )
        logger.info("Accuracy predictor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize accuracy predictor: {e}")
        return 1
    
    # Setup constraints
    constraints = {}
    if args.power_constraint is not None:
        constraints['power'] = args.power_constraint
    
    # Initialize evolution finder
    logger.info("Initializing evolution finder...")
    try:
        evolution_finder = CustomHardwareEvolutionFinder(
            ofa_network=ofa_network,
            latency_predictor=latency_predictor,
            accuracy_predictor=accuracy_predictor,
            latency_constraint=args.latency_constraint,
            accuracy_threshold=args.accuracy_threshold,
            quantization_scheme=args.quantization_scheme,
            objectives=args.objectives,
            constraints=constraints,
            population_size=args.population_size,
            max_iterations=args.max_iterations,
            mutation_prob=args.mutation_prob,
            crossover_prob=args.crossover_prob,
            elite_ratio=args.elite_ratio
        )
        logger.info("Evolution finder initialized")
    except Exception as e:
        logger.error(f"Failed to initialize evolution finder: {e}")
        return 1
    
    # Run search
    logger.info("Starting evolutionary search...")
    try:
        progress_save_path = None
        if args.save_progress:
            progress_dir = './search_progress'
            os.makedirs(progress_dir, exist_ok=True)
            progress_save_path = os.path.join(
                progress_dir,
                f'{args.hardware_name}_{args.quantization_scheme}_progress.json'
            )
        
        best_individuals, search_info = evolution_finder.run_evolution_search(
            verbose=args.verbose,
            save_progress=args.save_progress,
            progress_save_path=progress_save_path
        )
        logger.info("Search completed successfully")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return 1
    
    # Get Pareto front
    logger.info("Computing Pareto front...")
    try:
        pareto_front = evolution_finder.get_pareto_front()
        logger.info(f"Found {len(pareto_front)} Pareto optimal solutions")
    except Exception as e:
        logger.warning(f"Failed to compute Pareto front: {e}")
        pareto_front = []
    
    # Prepare results
    results = {
        'search_parameters': {
            'hardware_name': args.hardware_name,
            'ofa_network': args.ofa_network,
            'quantization_scheme': args.quantization_scheme,
            'latency_constraint': args.latency_constraint,
            'accuracy_threshold': args.accuracy_threshold,
            'objectives': args.objectives,
            'constraints': constraints,
            'population_size': args.population_size,
            'max_iterations': args.max_iterations
        },
        'search_info': search_info,
        'best_individuals': best_individuals,
        'pareto_front': pareto_front,
        'summary': {
            'total_runtime_s': search_info.get('search_duration', 0),
            'best_config': search_info.get('best_config'),
            'best_accuracy': search_info.get('best_evaluation', {}).get('accuracy', 0),
            'best_latency': search_info.get('best_evaluation', {}).get('latency', 0),
            'num_pareto_solutions': len(pareto_front)
        }
    }
    
    # Save results
    if args.output_path:
        output_path = args.output_path
    else:
        output_dir = './search_results'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f'{args.hardware_name}_{args.quantization_scheme}_{args.latency_constraint}ms.json'
        )
    
    try:
        save_results(results, output_path)
        logger.info(f"Results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return 1
    
    # Print summary
    best_eval = search_info.get('best_evaluation')
    if best_eval:
        logger.info("=" * 50)
        logger.info("SEARCH SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Best accuracy: {best_eval.get('accuracy', 0):.2f}%")
        logger.info(f"Best latency: {best_eval.get('latency', 0):.2f}ms")
        logger.info(f"Constraints met: {best_eval.get('constraints_met', False)}")
        logger.info(f"Search duration: {search_info.get('search_duration', 0):.2f}s")
        logger.info(f"Pareto solutions: {len(pareto_front)}")
        logger.info("=" * 50)
    
    logger.info("Search completed successfully!")
    return 0


if __name__ == '__main__':
    exit(main())
