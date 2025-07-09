#!/usr/bin/env python3
"""
Visualization Script for OFA Data Collection Results

This script creates visualizations of the collected accuracy and latency data,
showing the trade-offs between different subnet configurations.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Any, Tuple
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class OFADataVisualizer:
    """Visualizer for OFA data collection results."""
    
    def __init__(self, data_path: str, output_dir: str = "./plots"):
        """Initialize the visualizer."""
        self.data_path = data_path
        self.output_dir = output_dir
        self.data = self._load_data()
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process data
        self._process_data()
        
        logger.info(f"Loaded {len(self.data)} data points")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return []
    
    def _process_data(self):
        """Process and extract relevant metrics from data."""
        self.accuracies = []
        self.latencies = []
        self.model_sizes = []
        self.flops = []
        self.configs = []
        self.resolutions = []
        self.depths = []
        self.kernel_sizes = []
        self.expand_ratios = []
        
        for item in self.data:
            if item['accuracy'] is not None and item['latency'] is not None:
                self.accuracies.append(item['accuracy'])
                self.latencies.append(item['latency'])
                self.configs.append(item['config'])
                self.resolutions.append(item['config']['r'])
                self.depths.append(np.mean(item['config']['d']))
                self.kernel_sizes.append(np.mean(item['config']['ks']))
                self.expand_ratios.append(np.mean(item['config']['e']))
                
                # Calculate approximate model size and FLOPs
                model_size = self._calculate_model_size(item['config'])
                flops = self._calculate_flops(item['config'])
                
                self.model_sizes.append(model_size)
                self.flops.append(flops)
        
        # Convert to numpy arrays
        self.accuracies = np.array(self.accuracies)
        self.latencies = np.array(self.latencies)
        self.model_sizes = np.array(self.model_sizes)
        self.flops = np.array(self.flops)
        self.resolutions = np.array(self.resolutions)
        self.depths = np.array(self.depths)
        self.kernel_sizes = np.array(self.kernel_sizes)
        self.expand_ratios = np.array(self.expand_ratios)
        
        logger.info(f"Data processing complete:")
        logger.info(f"  Accuracy range: [{np.min(self.accuracies):.3f}, {np.max(self.accuracies):.3f}]")
        logger.info(f"  Latency range: [{np.min(self.latencies):.1f}, {np.max(self.latencies):.1f}] ms")
        logger.info(f"  Model size range: [{np.min(self.model_sizes):.1f}, {np.max(self.model_sizes):.1f}] M params")
        logger.info(f"  FLOPs range: [{np.min(self.flops):.1f}, {np.max(self.flops):.1f}] M FLOPs")
    
    def _calculate_model_size(self, config: Dict[str, Any]) -> float:
        """Calculate approximate model size in millions of parameters."""
        # Simplified calculation based on MobileNetV3 architecture
        # This is an approximation - actual calculation would require model analysis
        
        base_params = 5.0  # Base parameters (first/last layers)
        
        # Calculate parameters for each stage
        stage_params = 0
        for i, depth in enumerate(config['d']):
            # Average expand ratio and kernel size for this stage
            avg_expand = np.mean([config['e'][j] for j in range(i*4, (i+1)*4)])
            avg_kernel = np.mean([config['ks'][j] for j in range(i*4, (i+1)*4)])
            
            # Simplified parameter calculation per block
            block_params = 0.1 * avg_expand * (avg_kernel / 3) * depth
            stage_params += block_params
        
        # Resolution factor
        resolution_factor = (config['r'] / 224) ** 2
        
        total_params = (base_params + stage_params) * resolution_factor
        return total_params
    
    def _calculate_flops(self, config: Dict[str, Any]) -> float:
        """Calculate approximate FLOPs in millions."""
        # Simplified FLOPs calculation based on MobileNetV3
        
        resolution = config['r']
        base_flops = 20.0  # Base FLOPs for first/last layers
        
        # Calculate FLOPs for each stage
        stage_flops = 0
        current_resolution = resolution
        
        for i, depth in enumerate(config['d']):
            # Average expand ratio and kernel size for this stage
            avg_expand = np.mean([config['e'][j] for j in range(i*4, (i+1)*4)])
            avg_kernel = np.mean([config['ks'][j] for j in range(i*4, (i+1)*4)])
            
            # Simplified FLOPs calculation per block
            block_flops = 0.5 * avg_expand * (avg_kernel / 3) * (current_resolution ** 2) * depth
            stage_flops += block_flops
            
            # Resolution decreases with stride
            if i < len(config['d']) - 1:
                current_resolution = current_resolution // 2
        
        total_flops = base_flops + stage_flops
        return total_flops
    
    def create_accuracy_latency_plot(self, use_flops: bool = True, save_path: str = None):
        """Create accuracy vs latency scatter plot with model size as circle size."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use either FLOPs or parameter count for circle size
        sizes = self.flops if use_flops else self.model_sizes
        size_label = "FLOPs (M)" if use_flops else "Parameters (M)"
        
        # Normalize sizes for better visualization
        normalized_sizes = (sizes - np.min(sizes)) / (np.max(sizes) - np.min(sizes))
        circle_sizes = 20 + normalized_sizes * 300  # Scale to reasonable circle sizes
        
        # Create scatter plot
        scatter = ax.scatter(
            self.latencies, 
            self.accuracies,
            s=circle_sizes,
            c=self.resolutions,
            cmap='viridis',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add colorbar for resolution
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Input Resolution', rotation=270, labelpad=20)
        
        # Add labels and title
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('OFA Subnet Performance: Accuracy vs Latency\n(Circle size = Model Complexity)', fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend for circle sizes
        legend_sizes = [np.min(sizes), np.median(sizes), np.max(sizes)]
        legend_circles = []
        legend_labels = []
        
        for size in legend_sizes:
            norm_size = (size - np.min(sizes)) / (np.max(sizes) - np.min(sizes))
            circle_size = 20 + norm_size * 300
            legend_circles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor='gray', markersize=np.sqrt(circle_size/20),
                                           alpha=0.6, markeredgecolor='black'))
            legend_labels.append(f'{size:.1f}M')
        
        legend1 = ax.legend(legend_circles, legend_labels, 
                           title=size_label, loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.add_artist(legend1)
        
        # Add Pareto frontier
        pareto_indices = self._find_pareto_frontier()
        if len(pareto_indices) > 0:
            pareto_latencies = self.latencies[pareto_indices]
            pareto_accuracies = self.accuracies[pareto_indices]
            
            # Sort by latency for line plot
            sort_idx = np.argsort(pareto_latencies)
            ax.plot(pareto_latencies[sort_idx], pareto_accuracies[sort_idx], 
                   'r--', linewidth=2, alpha=0.7, label='Pareto Frontier')
            ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved accuracy vs latency plot to {save_path}")
        
        plt.show()
        return fig
    
    def _find_pareto_frontier(self) -> np.ndarray:
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
    
    def create_configuration_analysis(self, save_path: str = None):
        """Create subplot analysis of different configuration parameters."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Depth vs Accuracy
        axes[0, 0].scatter(self.depths, self.accuracies, alpha=0.6, s=50)
        axes[0, 0].set_xlabel('Average Depth')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Depth vs Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Kernel Size vs Latency
        axes[0, 1].scatter(self.kernel_sizes, self.latencies, alpha=0.6, s=50, color='orange')
        axes[0, 1].set_xlabel('Average Kernel Size')
        axes[0, 1].set_ylabel('Latency (ms)')
        axes[0, 1].set_title('Kernel Size vs Latency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Expand Ratio vs Model Size
        axes[1, 0].scatter(self.expand_ratios, self.model_sizes, alpha=0.6, s=50, color='green')
        axes[1, 0].set_xlabel('Average Expand Ratio')
        axes[1, 0].set_ylabel('Model Size (M params)')
        axes[1, 0].set_title('Expand Ratio vs Model Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Resolution vs FLOPs
        axes[1, 1].scatter(self.resolutions, self.flops, alpha=0.6, s=50, color='red')
        axes[1, 1].set_xlabel('Input Resolution')
        axes[1, 1].set_ylabel('FLOPs (M)')
        axes[1, 1].set_title('Resolution vs FLOPs')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved configuration analysis to {save_path}")
        
        plt.show()
        return fig
    
    def create_efficiency_frontier(self, save_path: str = None):
        """Create efficiency frontier plot showing trade-offs."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create efficiency metric (accuracy / latency)
        efficiency = self.accuracies / self.latencies
        
        # Color by efficiency
        scatter = ax.scatter(
            self.latencies,
            self.accuracies,
            c=efficiency,
            s=self.flops * 2,  # Size by FLOPs
            cmap='RdYlGn',
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Efficiency (Accuracy/Latency)', rotation=270, labelpad=20)
        
        # Labels and title
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Efficiency Frontier: Accuracy vs Latency\n(Circle size = FLOPs, Color = Efficiency)', fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Highlight top efficient models
        top_efficient_idx = np.argsort(efficiency)[-10:]  # Top 10 efficient models
        ax.scatter(
            self.latencies[top_efficient_idx],
            self.accuracies[top_efficient_idx],
            s=200,
            facecolors='none',
            edgecolors='red',
            linewidth=2,
            label='Top 10 Efficient Models'
        )
        
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved efficiency frontier to {save_path}")
        
        plt.show()
        return fig
    
    def create_summary_statistics(self, save_path: str = None):
        """Create summary statistics visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Accuracy distribution
        axes[0, 0].hist(self.accuracies, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Accuracy Distribution')
        axes[0, 0].axvline(np.mean(self.accuracies), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(self.accuracies):.3f}')
        axes[0, 0].legend()
        
        # 2. Latency distribution
        axes[0, 1].hist(self.latencies, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_xlabel('Latency (ms)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Latency Distribution')
        axes[0, 1].axvline(np.mean(self.latencies), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(self.latencies):.1f}ms')
        axes[0, 1].legend()
        
        # 3. Model size distribution
        axes[0, 2].hist(self.model_sizes, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_xlabel('Model Size (M params)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Model Size Distribution')
        axes[0, 2].axvline(np.mean(self.model_sizes), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(self.model_sizes):.1f}M')
        axes[0, 2].legend()
        
        # 4. FLOPs distribution
        axes[1, 0].hist(self.flops, bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 0].set_xlabel('FLOPs (M)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('FLOPs Distribution')
        axes[1, 0].axvline(np.mean(self.flops), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(self.flops):.1f}M')
        axes[1, 0].legend()
        
        # 5. Correlation matrix
        corr_data = np.column_stack([
            self.accuracies, self.latencies, self.model_sizes, self.flops,
            self.depths, self.kernel_sizes, self.expand_ratios, self.resolutions
        ])
        corr_matrix = np.corrcoef(corr_data.T)
        
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 1].set_title('Correlation Matrix')
        
        labels = ['Accuracy', 'Latency', 'Model Size', 'FLOPs', 
                 'Depth', 'Kernel Size', 'Expand Ratio', 'Resolution']
        axes[1, 1].set_xticks(range(len(labels)))
        axes[1, 1].set_yticks(range(len(labels)))
        axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, 1].set_yticklabels(labels)
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                axes[1, 1].text(j, i, f'{corr_matrix[i, j]:.2f}', 
                               ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 1])
        
        # 6. Efficiency vs Model Size
        efficiency = self.accuracies / self.latencies
        axes[1, 2].scatter(self.model_sizes, efficiency, alpha=0.6, s=50, color='purple')
        axes[1, 2].set_xlabel('Model Size (M params)')
        axes[1, 2].set_ylabel('Efficiency (Accuracy/Latency)')
        axes[1, 2].set_title('Efficiency vs Model Size')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved summary statistics to {save_path}")
        
        plt.show()
        return fig
    
    def generate_all_plots(self):
        """Generate all visualization plots."""
        logger.info("Generating all visualization plots...")
        
        # 1. Main accuracy vs latency plot
        self.create_accuracy_latency_plot(
            use_flops=True, 
            save_path=os.path.join(self.output_dir, 'accuracy_vs_latency_flops.png')
        )
        
        # 2. Accuracy vs latency with parameter count
        self.create_accuracy_latency_plot(
            use_flops=False,
            save_path=os.path.join(self.output_dir, 'accuracy_vs_latency_params.png')
        )
        
        # 3. Configuration analysis
        self.create_configuration_analysis(
            save_path=os.path.join(self.output_dir, 'configuration_analysis.png')
        )
        
        # 4. Efficiency frontier
        self.create_efficiency_frontier(
            save_path=os.path.join(self.output_dir, 'efficiency_frontier.png')
        )
        
        # 5. Summary statistics
        self.create_summary_statistics(
            save_path=os.path.join(self.output_dir, 'summary_statistics.png')
        )
        
        logger.info(f"All plots saved to {self.output_dir}")
    
    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("OFA DATA COLLECTION SUMMARY")
        print("="*60)
        print(f"Total data points: {len(self.data)}")
        print(f"Valid data points: {len(self.accuracies)}")
        print()
        print("PERFORMANCE METRICS:")
        print(f"  Accuracy: {np.mean(self.accuracies):.3f} ± {np.std(self.accuracies):.3f}")
        print(f"  Latency:  {np.mean(self.latencies):.1f} ± {np.std(self.latencies):.1f} ms")
        print(f"  Model Size: {np.mean(self.model_sizes):.1f} ± {np.std(self.model_sizes):.1f} M params")
        print(f"  FLOPs:    {np.mean(self.flops):.1f} ± {np.std(self.flops):.1f} M FLOPs")
        print()
        print("CONFIGURATION RANGES:")
        print(f"  Depth:        {np.min(self.depths):.1f} - {np.max(self.depths):.1f}")
        print(f"  Kernel Size:  {np.min(self.kernel_sizes):.1f} - {np.max(self.kernel_sizes):.1f}")
        print(f"  Expand Ratio: {np.min(self.expand_ratios):.1f} - {np.max(self.expand_ratios):.1f}")
        print(f"  Resolution:   {np.min(self.resolutions):.0f} - {np.max(self.resolutions):.0f}")
        print()
        
        # Find best models
        pareto_indices = self._find_pareto_frontier()
        if len(pareto_indices) > 0:
            print(f"PARETO FRONTIER: {len(pareto_indices)} models")
            
            # Best accuracy
            best_acc_idx = np.argmax(self.accuracies)
            print(f"  Best Accuracy: {self.accuracies[best_acc_idx]:.3f} @ {self.latencies[best_acc_idx]:.1f}ms")
            
            # Best latency
            best_lat_idx = np.argmin(self.latencies)
            print(f"  Best Latency:  {self.latencies[best_lat_idx]:.1f}ms @ {self.accuracies[best_lat_idx]:.3f} acc")
            
            # Best efficiency
            efficiency = self.accuracies / self.latencies
            best_eff_idx = np.argmax(efficiency)
            print(f"  Best Efficiency: {efficiency[best_eff_idx]:.4f} @ {self.accuracies[best_eff_idx]:.3f} acc, {self.latencies[best_eff_idx]:.1f}ms")
        
        print("="*60)


def main():
    """Main function to run visualization."""
    parser = argparse.ArgumentParser(description="Visualize OFA data collection results")
    parser.add_argument(
        "--data", 
        default="./data/collected/complete_dataset.json",
        help="Path to collected dataset JSON file"
    )
    parser.add_argument(
        "--output", 
        default="./plots",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--summary", 
        action="store_true",
        help="Print summary statistics only"
    )
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        print("Please run data collection first:")
        print("  python collect_data.py --config configs/stx_npu_config.yaml --samples 100")
        return
    
    # Initialize visualizer
    visualizer = OFADataVisualizer(args.data, args.output)
    
    # Print summary
    visualizer.print_summary()
    
    if not args.summary:
        # Generate all plots
        visualizer.generate_all_plots()
        print(f"\nAll visualizations saved to: {args.output}")


if __name__ == "__main__":
    main()
