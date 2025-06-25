"""
Custom Hardware Evolution Finder

Evolutionary search for optimal OFA sub-networks on custom NPU hardware
with quantization support.
"""

import copy
import random
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm
import logging

from .quantized_latency_predictor import QuantizedLatencyPredictor
from .quantized_accuracy_predictor import QuantizedAccuracyPredictor
from .utils import validate_subnet_config, format_subnet_config, save_results

logger = logging.getLogger(__name__)


class CustomHardwareArchManager:
    """Architecture manager for custom hardware evolution search."""
    
    def __init__(
        self,
        kernel_sizes: List[int] = [3, 5, 7],
        expand_ratios: List[int] = [3, 4, 6],
        depths: List[int] = [2, 3, 4],
        resolutions: List[int] = [160, 176, 192, 208, 224]
    ):
        self.num_blocks = 20
        self.num_stages = 5
        self.kernel_sizes = kernel_sizes
        self.expand_ratios = expand_ratios
        self.depths = depths
        self.resolutions = resolutions
    
    def random_sample(self) -> Dict[str, Any]:
        """Generate a random subnet configuration."""
        sample = {
            'ks': [random.choice(self.kernel_sizes) for _ in range(self.num_blocks)],
            'e': [random.choice(self.expand_ratios) for _ in range(self.num_blocks)],
            'd': [random.choice(self.depths) for _ in range(self.num_stages)],
            'r': [random.choice(self.resolutions)]
        }
        return sample
    
    def mutate_sample(self, sample: Dict[str, Any], mutation_prob: float = 0.1) -> Dict[str, Any]:
        """Mutate a subnet configuration."""
        mutated = copy.deepcopy(sample)
        
        # Mutate kernel sizes
        for i in range(self.num_blocks):
            if random.random() < mutation_prob:
                mutated['ks'][i] = random.choice(self.kernel_sizes)
        
        # Mutate expand ratios
        for i in range(self.num_blocks):
            if random.random() < mutation_prob:
                mutated['e'][i] = random.choice(self.expand_ratios)
        
        # Mutate depths
        for i in range(self.num_stages):
            if random.random() < mutation_prob:
                mutated['d'][i] = random.choice(self.depths)
        
        # Mutate resolution
        if random.random() < mutation_prob:
            mutated['r'][0] = random.choice(self.resolutions)
        
        return mutated
    
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Create offspring through crossover."""
        offspring = copy.deepcopy(parent1)
        
        # Crossover kernel sizes
        for i in range(self.num_blocks):
            if random.random() < 0.5:
                offspring['ks'][i] = parent2['ks'][i]
        
        # Crossover expand ratios
        for i in range(self.num_blocks):
            if random.random() < 0.5:
                offspring['e'][i] = parent2['e'][i]
        
        # Crossover depths
        for i in range(self.num_stages):
            if random.random() < 0.5:
                offspring['d'][i] = parent2['d'][i]
        
        # Crossover resolution
        if random.random() < 0.5:
            offspring['r'][0] = parent2['r'][0]
        
        return offspring


class CustomHardwareEvolutionFinder:
    """
    Evolutionary search for optimal OFA sub-networks on custom hardware.
    
    Supports multi-objective optimization with constraints on latency,
    accuracy, and hardware-specific metrics.
    """
    
    def __init__(
        self,
        ofa_network,
        latency_predictor: QuantizedLatencyPredictor,
        accuracy_predictor: QuantizedAccuracyPredictor,
        latency_constraint: float,
        accuracy_threshold: Optional[float] = None,
        quantization_scheme: str = 'int8',
        objectives: List[str] = ['accuracy'],
        constraints: Optional[Dict[str, float]] = None,
        population_size: int = 100,
        max_iterations: int = 500,
        mutation_prob: float = 0.1,
        crossover_prob: float = 0.8,
        elite_ratio: float = 0.2
    ):
        """
        Initialize the evolution finder.
        
        Args:
            ofa_network: OFA network instance
            latency_predictor: Quantized latency predictor
            accuracy_predictor: Quantized accuracy predictor
            latency_constraint: Maximum allowed latency (ms)
            accuracy_threshold: Minimum required accuracy (%)
            quantization_scheme: Quantization scheme
            objectives: List of objectives to optimize ['accuracy', 'latency', 'efficiency']
            constraints: Additional constraints dictionary
            population_size: Size of the population
            max_iterations: Maximum number of iterations
            mutation_prob: Mutation probability
            crossover_prob: Crossover probability
            elite_ratio: Ratio of elite individuals to preserve
        """
        self.ofa_network = ofa_network
        self.latency_predictor = latency_predictor
        self.accuracy_predictor = accuracy_predictor
        self.latency_constraint = latency_constraint
        self.accuracy_threshold = accuracy_threshold
        self.quantization_scheme = quantization_scheme
        self.objectives = objectives
        self.constraints = constraints or {}
        
        # Evolution parameters
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.elite_ratio = elite_ratio
        self.elite_size = int(population_size * elite_ratio)
        
        # Architecture manager
        self.arch_manager = CustomHardwareArchManager()
        
        # Evolution tracking
        self.population = []
        self.fitness_history = []
        self.best_individuals = []
        
    def _evaluate_individual(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate an individual subnet configuration.
        
        Args:
            config: Subnet configuration
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not validate_subnet_config(config):
            return {'fitness': float('-inf'), 'valid': False}
        
        try:
            # Predict latency and accuracy
            latency = self.latency_predictor.predict_latency(config)
            accuracy = self.accuracy_predictor.predict_single(config)
            
            # Check constraints
            constraints_met = True
            constraint_violations = {}
            
            # Latency constraint
            if latency > self.latency_constraint:
                constraints_met = False
                constraint_violations['latency'] = latency - self.latency_constraint
            
            # Accuracy threshold
            if self.accuracy_threshold and accuracy < self.accuracy_threshold:
                constraints_met = False
                constraint_violations['accuracy'] = self.accuracy_threshold - accuracy
            
            # Additional constraints
            for constraint_name, constraint_value in self.constraints.items():
                if constraint_name == 'power':
                    # Estimate power consumption (simplified)
                    power = self._estimate_power_consumption(config, latency)
                    if power > constraint_value:
                        constraints_met = False
                        constraint_violations['power'] = power - constraint_value
            
            # Calculate fitness
            fitness = self._calculate_fitness(config, accuracy, latency, constraints_met)
            
            return {
                'fitness': fitness,
                'accuracy': accuracy,
                'latency': latency,
                'constraints_met': constraints_met,
                'constraint_violations': constraint_violations,
                'valid': True
            }
            
        except Exception as e:
            logger.error(f"Error evaluating individual: {e}")
            return {'fitness': float('-inf'), 'valid': False}
    
    def _calculate_fitness(
        self, 
        config: Dict[str, Any], 
        accuracy: float, 
        latency: float, 
        constraints_met: bool
    ) -> float:
        """
        Calculate fitness score for an individual.
        
        Args:
            config: Subnet configuration
            accuracy: Predicted accuracy
            latency: Predicted latency
            constraints_met: Whether constraints are satisfied
            
        Returns:
            Fitness score
        """
        if not constraints_met:
            return float('-inf')  # Invalid solutions get very low fitness
        
        fitness = 0.0
        
        if 'accuracy' in self.objectives:
            # Maximize accuracy
            fitness += accuracy
        
        if 'latency' in self.objectives:
            # Minimize latency (convert to maximization)
            if latency > 0:
                fitness += (1000.0 / latency)  # Higher is better
        
        if 'efficiency' in self.objectives:
            # Maximize accuracy per unit latency
            if latency > 0:
                efficiency = accuracy / latency
                fitness += efficiency * 10  # Scale factor
        
        if 'hardware_efficiency' in self.objectives:
            # Use hardware-specific efficiency score
            hw_efficiency = self.latency_predictor.get_hardware_efficiency_score(config)
            fitness += hw_efficiency * 100  # Scale factor
        
        return fitness
    
    def _estimate_power_consumption(self, config: Dict[str, Any], latency: float) -> float:
        """
        Estimate power consumption for a subnet configuration.
        
        Args:
            config: Subnet configuration
            latency: Predicted latency
            
        Returns:
            Estimated power consumption (mW)
        """
        # Simplified power estimation
        # In practice, this should be based on hardware profiling
        
        # Base power consumption
        base_power = 100.0  # mW
        
        # Power scales with computation complexity
        complexity_factor = np.mean(config.get('e', [6])) * np.mean(config.get('ks', [7]))
        dynamic_power = complexity_factor * 10.0
        
        # Power scales with latency (longer execution = more energy)
        latency_power = latency * 0.1
        
        total_power = base_power + dynamic_power + latency_power
        return total_power
    
    def _initialize_population(self) -> List[Dict[str, Any]]:
        """Initialize the population with random individuals."""
        population = []
        
        for _ in range(self.population_size):
            individual = self.arch_manager.random_sample()
            population.append(individual)
        
        return population
    
    def _selection(self, population: List[Dict[str, Any]], fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """
        Select individuals for reproduction using tournament selection.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each individual
            
        Returns:
            Selected individuals
        """
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Tournament selection
            tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(copy.deepcopy(population[winner_idx]))
        
        return selected
    
    def _create_offspring(self, selected_population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create offspring through crossover and mutation.
        
        Args:
            selected_population: Selected parent population
            
        Returns:
            Offspring population
        """
        offspring = []
        
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[(i + 1) % len(selected_population)]
            
            if random.random() < self.crossover_prob:
                # Crossover
                child1 = self.arch_manager.crossover(parent1, parent2)
                child2 = self.arch_manager.crossover(parent2, parent1)
            else:
                # No crossover, just copy parents
                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)
            
            # Mutation
            child1 = self.arch_manager.mutate_sample(child1, self.mutation_prob)
            child2 = self.arch_manager.mutate_sample(child2, self.mutation_prob)
            
            offspring.extend([child1, child2])
        
        # Trim to original population size
        return offspring[:self.population_size]
    
    def run_evolution_search(
        self, 
        verbose: bool = True,
        save_progress: bool = True,
        progress_save_path: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run the evolutionary search.
        
        Args:
            verbose: Whether to print progress
            save_progress: Whether to save progress periodically
            progress_save_path: Path to save progress
            
        Returns:
            Tuple of (best_individuals, search_info)
        """
        logger.info(f"Starting evolution search with {self.population_size} individuals for {self.max_iterations} iterations")
        
        start_time = time.time()
        
        # Initialize population
        self.population = self._initialize_population()
        
        # Evolution loop
        for iteration in range(self.max_iterations):
            # Evaluate population
            evaluations = []
            fitness_scores = []
            
            for individual in self.population:
                evaluation = self._evaluate_individual(individual)
                evaluations.append(evaluation)
                fitness_scores.append(evaluation['fitness'])
            
            # Track best individuals
            best_idx = np.argmax(fitness_scores)
            best_individual = {
                'config': copy.deepcopy(self.population[best_idx]),
                'evaluation': evaluations[best_idx],
                'iteration': iteration
            }
            self.best_individuals.append(best_individual)
            
            # Track fitness history
            self.fitness_history.append({
                'iteration': iteration,
                'best_fitness': fitness_scores[best_idx],
                'avg_fitness': np.mean([f for f in fitness_scores if f != float('-inf')]),
                'num_valid': sum(1 for e in evaluations if e['valid']),
                'constraints_met': sum(1 for e in evaluations if e.get('constraints_met', False))
            })
            
            if verbose:
                logger.info(f"Iteration {iteration+1}/{self.max_iterations}: "
                           f"Best fitness={fitness_scores[best_idx]:.3f}, "
                           f"Best accuracy={evaluations[best_idx].get('accuracy', 0):.2f}%, "
                           f"Best latency={evaluations[best_idx].get('latency', 0):.2f}ms")
            
            # Save progress periodically
            if save_progress and (iteration + 1) % 50 == 0 and progress_save_path:
                self._save_progress(progress_save_path, iteration)
            
            # Selection
            selected_population = self._selection(self.population, fitness_scores)
            
            # Elite preservation
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elite_population = [copy.deepcopy(self.population[i]) for i in elite_indices]
            
            # Create offspring
            offspring = self._create_offspring(selected_population)
            
            # Replace population (keep elites)
            self.population = elite_population + offspring[:-self.elite_size]
        
        end_time = time.time()
        search_duration = end_time - start_time
        
        # Final evaluation and results
        final_evaluations = []
        final_fitness_scores = []
        
        for individual in self.population:
            evaluation = self._evaluate_individual(individual)
            final_evaluations.append(evaluation)
            final_fitness_scores.append(evaluation['fitness'])
        
        # Get best results
        valid_indices = [i for i, e in enumerate(final_evaluations) if e['valid'] and e.get('constraints_met', False)]
        
        if valid_indices:
            best_idx = max(valid_indices, key=lambda i: final_fitness_scores[i])
            best_config = self.population[best_idx]
            best_evaluation = final_evaluations[best_idx]
        else:
            logger.warning("No valid solutions found that meet constraints")
            best_config = None
            best_evaluation = None
        
        # Prepare search info
        search_info = {
            'search_duration': search_duration,
            'total_evaluations': len(self.population) * self.max_iterations,
            'best_config': best_config,
            'best_evaluation': best_evaluation,
            'fitness_history': self.fitness_history,
            'final_population_size': len(self.population),
            'valid_solutions': len(valid_indices),
            'search_parameters': {
                'population_size': self.population_size,
                'max_iterations': self.max_iterations,
                'mutation_prob': self.mutation_prob,
                'crossover_prob': self.crossover_prob,
                'latency_constraint': self.latency_constraint,
                'accuracy_threshold': self.accuracy_threshold,
                'objectives': self.objectives,
                'quantization_scheme': self.quantization_scheme
            }
        }
        
        logger.info(f"Evolution search completed in {search_duration:.2f}s")
        if best_config:
            logger.info(f"Best solution: accuracy={best_evaluation['accuracy']:.2f}%, "
                       f"latency={best_evaluation['latency']:.2f}ms")
        
        return self.best_individuals, search_info
    
    def _save_progress(self, save_path: str, iteration: int):
        """Save evolution progress."""
        progress_data = {
            'iteration': iteration,
            'fitness_history': self.fitness_history,
            'best_individuals': self.best_individuals,
            'population_size': len(self.population),
            'search_parameters': {
                'latency_constraint': self.latency_constraint,
                'accuracy_threshold': self.accuracy_threshold,
                'quantization_scheme': self.quantization_scheme
            }
        }
        
        progress_file = save_path.replace('.json', f'_progress_iter_{iteration}.json')
        save_results(progress_data, progress_file)
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """
        Get Pareto optimal solutions from the final population.
        
        Returns:
            List of Pareto optimal solutions
        """
        # Evaluate final population
        evaluations = []
        for individual in self.population:
            evaluation = self._evaluate_individual(individual)
            if evaluation['valid'] and evaluation.get('constraints_met', False):
                evaluations.append({
                    'config': individual,
                    'accuracy': evaluation['accuracy'],
                    'latency': evaluation['latency']
                })
        
        if not evaluations:
            return []
        
        # Find Pareto front (maximize accuracy, minimize latency)
        pareto_front = []
        
        for i, candidate in enumerate(evaluations):
            is_dominated = False
            
            for j, other in enumerate(evaluations):
                if i != j:
                    # Check if other dominates candidate
                    if (other['accuracy'] >= candidate['accuracy'] and 
                        other['latency'] <= candidate['latency'] and
                        (other['accuracy'] > candidate['accuracy'] or other['latency'] < candidate['latency'])):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        return pareto_front
