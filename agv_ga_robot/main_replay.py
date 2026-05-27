"""
Main replay script - visualize best genome performance.
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
    config_path = "agv_ga_robot/config/config.yaml"
    map_path = "agv_ga_robot/maps/learning_maze_simple.json"
    genome_path = "training_results/best_genome.pkl"
    
    if not os.path.exists(genome_path):
        print(f"[Error] Genome not found: {genome_path}")
        print("[Info] Run training first: python -m agv_ga_robot.main_train")
        sys.exit(1)
    
    print("[Replay] Loading...")
    config = load_config(config_path)
    map_data = MapLoader.load_map(map_path)
    
    # Load genome
    trainer = GATrainer(config, map_data)
    genome = trainer.load_genome(genome_path)
    
    # Create environment and network
    env = AGVEnvironment(config, map_data)
    obs, _ = env.reset()
    nn = NeuralNetwork(config['NETWORK_ARCH'])
    nn.set_genome(genome)
    
    # Run episode
    print("[Replay] Running...")
    trajectory = []
    for step in range(config['EPISODE_STEPS']):
        action = nn.forward(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append((env.robot.x, env.robot.y))
        if terminated or truncated:
            print(f"[Replay] Ended at step {step + 1}")
            break
    
    # Statistics
    stats = env.episode_data.get_summary()
    print(f"[Replay] Checkpoints: {stats['checkpoints_visited']}")
    print(f"[Replay] Collisions: {stats['collision_count']}")
    print(f"[Replay] Reward: {stats['total_reward']:.0f}")
    
    # Visualize trajectory
    try:
        print("[Visualization] Displaying trajectory...")
        viz = SimpleVisualization(map_data, config)
        
        # Display the trajectory (as leader)
        viz.update_generation(0, stats['total_reward'], 0)
        viz.update_episode(trajectory, stats, [], [])
        
        # Keep window open until user closes it
        import pygame
        clock = pygame.time.Clock()
        running = True
        while running:
            running = viz.handle_events()
            clock.tick(30)
        
        viz.close()
    except Exception as e:
        print(f"[Visualization Error] {e}")

