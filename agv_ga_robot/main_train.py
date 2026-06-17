"""
Main training script with real-time visualization.

Runs GA training with live single-window visualization showing overlayed trajectories:
- Leader (best from previous generation) - green
- Top 3 from current generation - blue/cyan/magenta
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agv_ga_robot.utils.helpers import load_config
from agv_ga_robot.maps.map_loader import MapLoader
from agv_ga_robot.ga.ga_trainer import GATrainer
from agv_ga_robot.ui.visualization_simple import SimpleVisualization


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GA robot")
    parser.add_argument('--map', '-m', default='learning_maze_simple',
                        help='Map name (without path or .json)')
    parser.add_argument('--population', '-p', type=int, default=None,
                        help='Population size (default from config)')
    parser.add_argument('--generations', '-g', type=int, default=None,
                        help='Number of generations (default from config)')
    parser.add_argument('--config', '-c', default='agv_ga_robot/config/config.yaml',
                        help='Path to config YAML')
    args = parser.parse_args()
    
    # Configuration
    config_path = args.config
    config = load_config(config_path)
    
    # Override population and generations if provided
    if args.population is not None:
        config['GA_PARAMS']['population_size'] = args.population
    if args.generations is not None:
        config['GA_PARAMS']['num_generations'] = args.generations
    
    # Map path
    map_path = f"agv_ga_robot/maps/{args.map}.json"
    if not os.path.exists(map_path):
        print(f"[Error] Map not found: {map_path}")
        sys.exit(1)
    
    map_data = MapLoader.load_map(map_path)
    
    # Output directories
    output_dir = "training_results"
    final_output_dir = "training_results_final"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)
    
    print("[Training] Starting...")
    print(f"  Config: {config_path}")
    print(f"  Map: {args.map}")
    print(f"  Output (gen): {output_dir}")
    print(f"  Output (final): {final_output_dir}")
    print(f"  Population: {config['GA_PARAMS']['population_size']}")
    print(f"  Generations: {config['GA_PARAMS']['num_generations']}")
    print()
    
    # Initialize visualization
    try:
        viz = SimpleVisualization(map_data, config)
        print("[Visualization] Ready (pygame-based)")
    except Exception as e:
        print(f"[Warning] No visualization: {e}")
        viz = None
    
    # Train
    trainer = GATrainer(config, map_data, output_dir=output_dir, 
                       final_output_dir=final_output_dir, visualization=viz)
    best_genome, best_fitness = trainer.train(verbose=True)
    
    # Cleanup
    if viz:
        viz.close()
    
    print(f"\n[Done] Best fitness: {best_fitness:.2f}")
    print(f"[Done] Generation results in: {output_dir}/")
    print(f"[Done] Final genome in: {final_output_dir}/")
