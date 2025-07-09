#!/usr/bin/env python3
"""
Example script demonstrating how to use the OFA data visualizer.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_sample_data():
    """Create sample data for demonstration if no real data exists."""
    print("Creating sample data for demonstration...")
    
    # Create sample data that mimics real OFA data collection
    sample_data = []
    
    np.random.seed(42)  # For reproducible results
    
    for i in range(200):
        # Random subnet configuration
        config = {
            'ks': [np.random.choice([3, 5, 7]) for _ in range(20)],
            'e': [np.random.choice([3, 4, 6]) for _ in range(20)],
            'd': [np.random.choice([2, 3, 4]) for _ in range(5)],
            'r': np.random.choice([160, 176, 192, 208, 224])
        }
        
        # Calculate synthetic metrics based on configuration
        # Higher resolution, depth, expand ratio -> higher accuracy but higher latency
        complexity_factor = (
            np.mean(config['ks']) / 5.0 +
            np.mean(config['e']) / 4.5 +
            np.mean(config['d']) / 3.0 +
            config['r'] / 224.0
        ) / 4.0
        
        # Synthetic accuracy (70-85% range)
        base_accuracy = 0.70 + 0.15 * complexity_factor
        accuracy = base_accuracy + np.random.normal(0, 0.02)
        accuracy = np.clip(accuracy, 0.65, 0.90)
        
        # Synthetic latency (5-50ms range, inversely related to efficiency)
        base_latency = 5.0 + 45.0 * complexity_factor
        latency = base_latency + np.random.normal(0, 2.0)
        latency = np.clip(latency, 3.0, 60.0)
        
        data_point = {
            'config': config,
            'accuracy': accuracy,
            'latency': latency,
            'input_shape': [1, 3, config['r'], config['r']],
            'quantization': {
                'enabled': True,
                'mode': 'static',
                'weight_type': 'QInt8',
                'activation_type': 'QUInt8'
            },
            'hardware': 'CustomNPU',
            'timestamp': 1641234567.0 + i
        }
        
        sample_data.append(data_point)
    
    # Save sample data
    os.makedirs('./data/collected', exist_ok=True)
    with open('./data/collected/sample_dataset.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("Sample data created: ./data/collected/sample_dataset.json")
    return './data/collected/sample_dataset.json'

def main():
    """Main function to demonstrate visualization."""
    print("OFA Data Visualization Demo")
    print("="*40)
    
    # Check if real data exists
    real_data_path = './data/collected/complete_dataset.json'
    
    if os.path.exists(real_data_path):
        print(f"Found real data: {real_data_path}")
        data_path = real_data_path
    else:
        print(f"No real data found at {real_data_path}")
        data_path = create_sample_data()
    
    print(f"\nUsing data from: {data_path}")
    
    # Try to import and run visualizer
    try:
        from visualize_data import OFADataVisualizer
        
        # Initialize visualizer
        visualizer = OFADataVisualizer(data_path, "./plots")
        
        # Print summary
        visualizer.print_summary()
        
        # Generate sample plots
        print("\nGenerating visualization plots...")
        
        # Create main accuracy vs latency plot
        print("1. Creating accuracy vs latency plot...")
        visualizer.create_accuracy_latency_plot(
            use_flops=True,
            save_path="./plots/demo_accuracy_vs_latency.png"
        )
        
        # Create efficiency frontier
        print("2. Creating efficiency frontier...")
        visualizer.create_efficiency_frontier(
            save_path="./plots/demo_efficiency_frontier.png"
        )
        
        # Create configuration analysis
        print("3. Creating configuration analysis...")
        visualizer.create_configuration_analysis(
            save_path="./plots/demo_configuration_analysis.png"
        )
        
        print("\nVisualization complete!")
        print("Check the ./plots directory for generated images.")
        
    except ImportError as e:
        print(f"Error importing visualizer: {e}")
        print("Please install required packages:")
        print("  pip install matplotlib seaborn scikit-learn")
        
    except Exception as e:
        print(f"Error running visualization: {e}")
        print("Please check the error message above.")

if __name__ == "__main__":
    main()
