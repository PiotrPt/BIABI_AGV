"""GA Trainer - Genetic Algorithm trainer using PyGAD."""

import numpy as np
import pygad
from typing import Dict, Any, Callable, List, Tuple
import os
import pickle

from ..robot import NeuralNetwork
from ..env import AGVEnvironment
from ..utils import load_config
from .fitness import calculate_fitness, get_episode_stats_from_env


class GATrainer:
    """Wrapper around PyGAD for training AGV robot."""
    
    def __init__(self, config: Dict[str, Any], map_data: Dict[str, Any], 
                 output_dir: str = "results", final_output_dir: str = "training_results_final",
                 visualization=None):
        """
        Initialize GA trainer.
        
        Args:
            config: Configuration dictionary
            map_data: Map data for environment
            output_dir: Directory to save generation-based results (best_genome_gen*.pkl)
            final_output_dir: Directory to save final indexed results (best_genome_final_*.pkl)
            visualization: Optional Visualization object for real-time display
        """
        self.config = config
        self.map_data = map_data
        self.output_dir = output_dir
        self.final_output_dir = final_output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(final_output_dir, exist_ok=True)
        self.visualization = visualization
        
        # Extract GA parameters
        ga_cfg = config['GA_PARAMS']
        reward_cfg = config['REWARD_WEIGHTS']
        
        self.population_size = ga_cfg['population_size']
        self.num_generations = ga_cfg['num_generations']
        self.mutation_percent = ga_cfg['mutation_percent']
        self.crossover_type = ga_cfg['crossover_type']
        self.parent_selection = ga_cfg['parent_selection']
        self.tournament_size = ga_cfg['tournament_size']
        self.fitness_evaluations = ga_cfg.get('fitness_evaluations', 3)
        self.reward_weights = reward_cfg
        
        # Neural network template
        network_arch = config['NETWORK_ARCH']
        self.template_nn = NeuralNetwork(network_arch)
        self.genome_size = self.template_nn.get_genome_size()
        
        # Track best across generations for visualization
        self.prev_best_genome = None
        self.prev_best_trajectory = []
        self.prev_best_stats = {}
        
        print(f"[GA Trainer] Network architecture: {network_arch}")
        print(f"[GA Trainer] Genome size: {self.genome_size}")
        print(f"[GA Trainer] Population: {self.population_size}")
        print(f"[GA Trainer] Generations: {self.num_generations}")
        print(f"[GA Trainer] Fitness evaluations per genome: {self.fitness_evaluations}")
        
        # Tracking
        self.generation_stats = []
        self.best_genome = None
        self.best_fitness = -float('inf')
        self.ga_instance = None
        
        # Environment pool for evaluation
        self.env = None
    
    def _fitness_function(self, ga_instance, solution: np.ndarray, solution_idx: int) -> float:
        """
        Fitness function for PyGAD.
        Evaluates genome multiple times and returns average fitness.
        
        Args:
            ga_instance: PyGAD instance
            solution: Genome (flat array)
            solution_idx: Index in population
        
        Returns:
            Average fitness score
        """
        # Evaluate genome multiple times
        fitness_scores = []
        
        for eval_idx in range(self.fitness_evaluations):
            # Create environment if not exists
            if self.env is None:
                self.env = AGVEnvironment(self.config, self.map_data)
            
            # Set up network with genome
            nn = NeuralNetwork(self.config['NETWORK_ARCH'])
            nn.set_genome(solution)
            
            # Run episode
            obs, _ = self.env.reset(seed=eval_idx)  # Different seed per eval
            
            episode_reward = 0.0
            terminated = False
            
            while not terminated:
                # Get network prediction
                action = nn.forward(obs)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
            
            # Get fitness from episode
            episode_stats = get_episode_stats_from_env(self.env)
            fitness = calculate_fitness(episode_stats, self.reward_weights)
            fitness_scores.append(fitness)
        
        # Return average fitness
        avg_fitness = np.mean(fitness_scores)
        return avg_fitness
    
    def _on_generation(self, ga_instance) -> None:
        """Called at end of each generation."""
        gen = ga_instance.generations_completed
        best_fitness = ga_instance.best_solutions_fitness[-1]
        
        # Get population statistics
        population_fitness = ga_instance.last_generation_fitness
        avg_fitness = np.mean(population_fitness)
        worst_fitness = np.min(population_fitness)
        
        # Store stats
        stats = {
            'generation': gen,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'worst_fitness': worst_fitness,
        }
        self.generation_stats.append(stats)
        
        # Print progress only every 10 generations
        if gen % 10 == 0 or gen == 1:
            print(f"[Gen {gen:3d}] Best: {best_fitness:10.0f} | Avg: {avg_fitness:10.0f} | Worst: {worst_fitness:10.0f}")
        
        # Visualization callback
        if self.visualization is not None:
            try:
                # Get top-3 genomes from current generation
                # Use direct population access to avoid indexing issues
                try:
                    population = ga_instance.population
                    fitness = ga_instance.last_generation_fitness
                except:
                    # Fallback if population access fails
                    population = ga_instance.last_generation_parents
                    fitness = ga_instance.last_generation_fitness
                
                # Ensure we have valid fitness and population
                if len(fitness) < 3:
                    print(f"  [Warning] Not enough fitness values: {len(fitness)}")
                    return
                
                if len(population) < 3:
                    print(f"  [Warning] Not enough genomes: {len(population)}")
                    return
                
                top_indices = np.argsort(fitness)[::-1][:min(3, len(fitness))]
                top_genomes = [population[idx].copy() if isinstance(population[idx], np.ndarray) else population[idx] 
                              for idx in top_indices]
                
                # Generate trajectories for top-3 using FULL episode steps (not shortened)
                # This ensures visualization shows the same fitness as GA evaluation
                top3_data = []
                for i, genome in enumerate(top_genomes):
                    try:
                        # Use full episode steps - vizualizacja może pokazać tylko część
                        traj, obs, action, ep_stats = self._run_episode_for_genome(genome, 
                                                                                    max_steps=self.config['EPISODE_STEPS'])
                        top3_data.append({
                            'trajectory': traj if traj else [],
                            'obs': obs,
                            'action': action,
                            'stats': ep_stats
                        })
                    except Exception as e:
                        import traceback
                        error_msg = traceback.format_exc()
                        print(f"    [Error trajectory {i}]: {str(e)}")
                        top3_data.append({'trajectory': [], 'obs': np.zeros(16), 'action': np.array([0, 0]), 'stats': {}})
                
                # LEADER is RANK 1 from current generation
                leader_trajectory = top3_data[0]['trajectory'] if top3_data else []
                leader_stats = top3_data[0]['stats'] if top3_data else {}
                
                # top3_trajectories will have RANK 2-3 (or empty if < 3 genomes)
                top3_trajectories = []
                top3_stats = []
                # Add rank 2 (if exists)
                if len(top3_data) > 1:
                    top3_trajectories.append(top3_data[1]['trajectory'])
                    top3_stats.append(top3_data[1]['stats'])
                # Add rank 3 (if exists)  
                if len(top3_data) > 2:
                    top3_trajectories.append(top3_data[2]['trajectory'])
                    top3_stats.append(top3_data[2]['stats'])
                
                # Update visualization
                self.visualization.update_generation(gen, best_fitness, avg_fitness)
                self.visualization.update_episode(leader_trajectory, leader_stats,
                                                 top3_trajectories, top3_stats)
                
                # Handle window events
                if not self.visualization.handle_events():
                    print("\n[Visualization] Window closed by user")
                    ga_instance.stop_criteria_met = True
                
                # Save current best as next leader
                if top_genomes:
                    self.prev_best_genome = top_genomes[0].copy()
                    self.prev_best_trajectory = top3_data[0]['trajectory']
                    self.prev_best_stats = top3_data[0]['stats']
                    
            except Exception as e:
                print(f"[Visualization Error] {e}")
        
        # Save best genome periodically
        save_freq = self.config.get('SAVE_FREQUENCY', 10)
        if gen % save_freq == 0:
            # Prefer the confirmed best_genome, otherwise save the prev_best_genome from visualization
            genome_to_save = None
            if self.best_genome is not None:
                genome_to_save = self.best_genome
            elif hasattr(self, 'prev_best_genome') and self.prev_best_genome is not None:
                genome_to_save = self.prev_best_genome

            if genome_to_save is not None:
                self.save_best_genome(f"best_genome_gen{gen}.pkl", genome=genome_to_save)
            else:
                print(f"[Warning] No genome available to save at generation {gen}")
    
    def train(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Train the GA.
        
        Args:
            verbose: Print progress
        
        Returns:
            (best_genome, best_fitness)
        """
        print(f"\n{'='*60}")
        print("Starting GA Training")
        print(f"{'='*60}\n")
        
        # Initialize PyGAD
        self.ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=max(2, self.population_size // 3),
            fitness_func=self._fitness_function,
            sol_per_pop=self.population_size,
            num_genes=self.genome_size,
            init_range_low=-0.5,
            init_range_high=0.5,
            parent_selection_type=self.parent_selection,
            K_tournament=self.tournament_size,
            crossover_type=self.crossover_type,
            mutation_type="random",
            mutation_percent_genes=self.mutation_percent,
            on_generation=self._on_generation if verbose else None,
            random_seed=self.config.get('SEED', None),
            suppress_warnings=not verbose,
        )
        
        # Run GA
        print(f"Training for {self.num_generations} generations...")
        self.ga_instance.run()
        
        # Get best solution from final generation
        best_solution_result = self.ga_instance.best_solution()
        # best_solution() returns (solution, fitness, idx)
        if isinstance(best_solution_result, tuple):
            best_solution = best_solution_result[0]
        else:
            best_solution = best_solution_result
        
        best_fitness = self.ga_instance.best_solutions_fitness[-1]
        
        self.best_genome = best_solution.copy() if hasattr(best_solution, 'copy') else np.array(best_solution)
        self.best_fitness = best_fitness
        
        print(f"\n{'='*60}")
        print("Training Complete")
        print(f"{'='*60}")
        print(f"Best Fitness: {self.best_fitness:.2f}")
        print(f"Generations: {len(self.generation_stats)}")
        # Save final best genome
        try:
            self.save_best_genome("best_genome_final.pkl")
        except Exception:
            pass

        return self.best_genome, self.best_fitness
    
    def _find_next_final_index(self) -> int:
        """
        Find next index for final genome file.
        Searches for best_genome_final_*.pkl files and returns max_index + 1.
        
        Returns:
            Next index for final genome filename
        """
        import re
        max_idx = 0
        try:
            for fname in os.listdir(self.final_output_dir):
                match = re.match(r'best_genome_final_(\d+)\.pkl', fname)
                if match:
                    idx = int(match.group(1))
                    max_idx = max(max_idx, idx)
        except FileNotFoundError:
            pass
        return max_idx + 1
    
    def save_best_genome(self, filename: str, genome: np.ndarray = None) -> None:
        """
        Save best genome to file.
        
        Smart saving:
        - If filename contains 'gen': save to output_dir, overwrites previous gen file
        - If filename contains 'final': save to final_output_dir with auto-increment index
        
        Args:
            filename: Filename to save (e.g. 'best_genome_gen10.pkl' or 'best_genome_final.pkl')
            genome: Optional genome to save (defaults to self.best_genome)
        """
        # Allow saving a provided genome (e.g., best from current generation) or fall back to stored best_genome
        genome_to_write = genome if genome is not None else self.best_genome
        if genome_to_write is None:
            print("[Warning] No genome available to save")
            return
        
        # Determine output directory and filename based on pattern
        if 'final' in filename.lower():
            # Final genome - save with auto-increment index
            next_idx = self._find_next_final_index()
            # Extract basename without .pkl
            base = filename.replace('.pkl', '')
            final_filename = f"{base}_{next_idx}.pkl"
            filepath = os.path.join(self.final_output_dir, final_filename)
        else:
            # Generation genome - save to output_dir (will overwrite)
            filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(genome_to_write, f)
        print(f"[Saved] Genome to {filepath}")
    
    def load_genome(self, filepath: str) -> np.ndarray:
        """
        Load genome from file.
        
        Args:
            filepath: Path to genome file
        
        Returns:
            Genome array
        """
        with open(filepath, 'rb') as f:
            genome = pickle.load(f)
        print(f"[Loaded] Genome from {filepath}")
        return genome
    
    
    def get_top_n_genomes(self, n: int = 3) -> List[np.ndarray]:
        """
        Get top-N genomes from last generation.
        
        Args:
            n: Number of top genomes to return
        
        Returns:
            List of top-N genome arrays
        """
        if not hasattr(self, 'ga_instance') or self.ga_instance is None:
            print("[Warning] GA instance not available")
            return []
        
        # Get last generation population and fitness
        population = self.ga_instance.last_generation_parents
        fitness = self.ga_instance.last_generation_fitness
        
        # Sort by fitness (descending)
        top_indices = np.argsort(fitness)[::-1][:n]
        top_genomes = [population[idx].copy() for idx in top_indices]
        
        return top_genomes
    
    
    def _run_episode_for_genome(self, genome: np.ndarray, max_steps: int = 100) -> Tuple[List, np.ndarray, np.ndarray, Dict]:
        """
        Run a single episode with given genome for visualization.
        
        Args:
            genome: Genome to evaluate
            max_steps: Max steps for episode
        
        Returns:
            (trajectory, last_obs, last_action, stats_dict)
        """
        env = AGVEnvironment(self.config, self.map_data)
        obs, _ = env.reset()
        
        # Set up NN with genome
        nn = NeuralNetwork(self.config['NETWORK_ARCH'])
        nn.set_genome(genome)
        
        trajectory = []
        last_action = np.array([0, 0])
        
        for _ in range(max_steps):
            action = nn.forward(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            trajectory.append((env.robot.x, env.robot.y))
            last_action = action
            
            if terminated or truncated:
                break
        
        stats = env.episode_data.get_summary()
        stats['visited_checkpoints'] = [i for i, v in enumerate(env.visited_checkpoints) if v]
        
        return trajectory, obs, last_action, stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.generation_stats:
            return {}
        
        stats = np.array([s['best_fitness'] for s in self.generation_stats])
        avg_stats = np.array([s['avg_fitness'] for s in self.generation_stats])
        
        return {
            'total_generations': len(self.generation_stats),
            'best_fitness': self.best_fitness,
            'worst_fitness': float(np.min(stats)),
            'avg_best_fitness': float(np.mean(stats)),
            'final_avg_population_fitness': float(avg_stats[-1]) if len(avg_stats) > 0 else 0,
            'generation_stats': self.generation_stats,
        }
    
    def save_statistics(self, filename: str = "training_stats.pkl") -> None:
        """Save training statistics."""
        filepath = os.path.join(self.output_dir, filename)
        stats = self.get_statistics()
        with open(filepath, 'wb') as f:
            pickle.dump(stats, f)
        print(f"[Saved] Training statistics to {filepath}")
