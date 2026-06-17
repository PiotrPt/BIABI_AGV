"""
Main replay script - visualize best genome performance with animation.
"""

import os
import sys
import pickle
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agv_ga_robot.utils.helpers import load_config
from agv_ga_robot.maps.map_loader import MapLoader
from agv_ga_robot.ga.ga_trainer import GATrainer
from agv_ga_robot.env.agv_env import AGVEnvironment
from agv_ga_robot.robot.neural_network import NeuralNetwork
from agv_ga_robot.ui.visualization_simple import SimpleVisualization




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replay a trained genome")
    parser.add_argument('--genome', '-g', help='Path to genome pickle file')
    parser.add_argument('--map', '-m', default='learning_maze_simple',
                        help='Map name (without path or .json)')
    parser.add_argument('--config', '-c', default='agv_ga_robot/config/config.yaml',
                        help='Path to config YAML')
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent
    
    # Map path
    map_path = project_root / "agv_ga_robot" / "maps" / f"{args.map}.json"
    if not os.path.exists(map_path):
        print(f"[Error] Map not found: {map_path}")
        sys.exit(1)

    map_data = MapLoader.load_map(str(map_path))
    
    # Genome path
    genome_path = args.genome
    if not genome_path:
        # Try to find latest final genome
        final_dir = project_root / 'training_results_final'
        if final_dir.exists():
            final_files = []
            for file_path in final_dir.iterdir():
                if file_path.suffix == '.pkl' and file_path.name.startswith('best_genome_final_'):
                    try:
                        index = int(file_path.stem.rsplit('_', 1)[-1])
                    except ValueError:
                        index = -1
                    final_files.append((index, file_path))
            if final_files:
                genome_path = str(sorted(final_files, key=lambda item: item[0])[-1][1])
        
        if not genome_path:
            print(f"[Error] No genome path specified and no final genomes found")
            print("[Info] Specify with: python -m agv_ga_robot.main_replay --genome <path>")
            sys.exit(1)
    
    if not os.path.exists(genome_path):
        print(f"[Error] Genome not found: {genome_path}")
        print("[Info] Run training first: python main.py")
        sys.exit(1)

    print("[Replay] Loading...")
    print(f"  Config: {config_path}")
    print(f"  Map: {args.map}")
    print(f"  Genome: {genome_path}")
    
    # Load genome
    trainer = GATrainer(config, map_data)
    genome = trainer.load_genome(genome_path)
    
    # Create environment and network
    env = AGVEnvironment(config, map_data)
    obs, _ = env.reset()
    nn = NeuralNetwork(config['NETWORK_ARCH'])
    nn.set_genome(genome)
    
    # Run episode and collect trajectory with robot angles
    print("[Replay] Running episode...")
    trajectory = []
    robot_angles = []
    
    for step in range(config['EPISODE_STEPS']):
        action = nn.forward(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append((env.robot.x, env.robot.y))
        robot_angle = getattr(env.robot, 'angle_rad', None)
        if robot_angle is None:
            robot_angle = getattr(env.robot, 'angle', 0.0)
        robot_angles.append(robot_angle)
        
        if terminated or truncated:
            print(f"[Replay] Episode ended at step {step + 1}")
            break
    
    # Statistics
    stats = env.episode_data.get_summary()
    print(f"[Replay] Checkpoints: {stats['checkpoints_visited']}")
    print(f"[Replay] Collisions: {stats['collision_count']}")
    print(f"[Replay] Reward: {stats['total_reward']:.0f}")
    
    # Visualize trajectory with animation
    try:
        print("[Visualization] Starting animation...")
        viz = SimpleVisualization(map_data, config)
        
        # Run animation
        running = True
        while running:
            running = viz.draw_animated_trajectory(trajectory, robot_angles, stats, playback_speed=1.0)
        
        viz.close()
        print("[Visualization] Animation complete")
    except Exception as e:
        print(f"[Visualization Error] {e}")
        import traceback
        traceback.print_exc()

