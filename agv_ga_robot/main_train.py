"""
Main training script with real-time visualization.

Runs GA training with live single-window visualization showing overlayed trajectories:
- Leader (best from previous generation) - green
- Top 3 from current generation - blue/cyan/magenta
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agv_ga_robot.utils.helpers import load_config
from agv_ga_robot.maps.map_loader import MapLoader
from agv_ga_robot.ga.ga_trainer import GATrainer
from agv_ga_robot.ui.visualization import Visualization


if __name__ == "__main__":
    # Configuration
    config_path = "agv_ga_robot/config/config.yaml"
    map_path = "agv_ga_robot/maps/learning_maze_simple.json"
    output_dir = "training_results"
    
    # Load configuration
    config = load_config(config_path)
    map_data = MapLoader.load_map(map_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("[Training] Starting...")
    print(f"  Config: {config_path}")
    print(f"  Map: {map_path}")
    print(f"  Output: {output_dir}")
    print(f"  Population: {config['GA_PARAMS']['population_size']}")
    print(f"  Generations: {config['GA_PARAMS']['num_generations']}")
    print()
    
    # Initialize visualization
    try:
        viz = Visualization(map_data, config)
        print("[Visualization] Ready")
    except Exception as e:
        print(f"[Warning] No visualization: {e}")
        viz = None
    
    # Train
    trainer = GATrainer(config, map_data, output_dir=output_dir, visualization=viz)
    best_genome, best_fitness = trainer.train(verbose=True)
    
    # Cleanup
    if viz:
        viz.close()
    
    print(f"\n[Done] Best fitness: {best_fitness:.2f}")
    print(f"[Done] Results in: {output_dir}/")
