"""Test script for Phase 2 - GA Training."""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agv_ga_robot.utils import load_config
from agv_ga_robot.maps import MapLoader
from agv_ga_robot.robot import NeuralNetwork
from agv_ga_robot.env import AGVEnvironment
from agv_ga_robot.ga import GATrainer, calculate_fitness


def test_fitness_function():
    """Test fitness calculation."""
    print("\n" + "="*60)
    print("TEST 1: Fitness Function")
    print("="*60)
    
    script_dir = os.path.dirname(__file__)
    config = load_config(os.path.join(script_dir, "config", "config.yaml"))
    
    reward_weights = config['REWARD_WEIGHTS']
    
    # Test case 1: Perfect episode
    episode_data_perfect = {
        'checkpoints_visited': 3,
        'collision_count': 0,
        'total_distance_improvement': 500.0,
        'total_speed': 100.0,
        'stuck_count': 0,
        'steps': 100,
    }
    
    fitness_perfect = calculate_fitness(episode_data_perfect, reward_weights)
    print(f"Perfect episode fitness: {fitness_perfect:.2f}")
    
    # Test case 2: Poor episode
    episode_data_poor = {
        'checkpoints_visited': 0,
        'collision_count': 3,
        'total_distance_improvement': 0.0,
        'total_speed': 10.0,
        'stuck_count': 5,
        'steps': 100,
    }
    
    fitness_poor = calculate_fitness(episode_data_poor, reward_weights)
    print(f"Poor episode fitness: {fitness_poor:.2f}")
    
    # Fitness should be higher for perfect episode
    assert fitness_perfect > fitness_poor, "Perfect episode should have higher fitness"
    print("[OK] Fitness function test PASSED")


def test_neural_network_inference():
    """Test that neural network can be used for inference."""
    print("\n" + "="*60)
    print("TEST 2: Neural Network Inference")
    print("="*60)
    
    # Create network
    net = NeuralNetwork([16, 32, 16, 2])
    
    # Get initial genome
    genome = net.get_genome()
    print(f"Genome size: {len(genome)}")
    
    # Test inference
    test_obs = np.random.randn(16).astype(np.float32)
    output = net.forward(test_obs)
    
    print(f"Input: {test_obs[:3]}...")
    print(f"Output: {output}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Output should be in [-1, 1]
    assert np.all(output >= -1.0) and np.all(output <= 1.0), "Output should be in [-1, 1]"
    print("✓ Neural network inference test PASSED")


def test_environment_episode():
    """Test running a full episode."""
    print("\n" + "="*60)
    print("TEST 3: Environment Episode")
    print("="*60)
    
    script_dir = os.path.dirname(__file__)
    config = load_config(os.path.join(script_dir, "config", "config.yaml"))
    map_data = MapLoader.load_map(os.path.join(script_dir, "maps", "learning_maze_simple.json"))
    
    env = AGVEnvironment(config, map_data)
    nn = NeuralNetwork(config['NETWORK_ARCH'])
    
    print(f"Running episode with random network...")
    
    obs, _ = env.reset()
    total_reward = 0
    checkpoints = 0
    collisions = 0
    
    for step in range(100):  # Run 100 steps
        action = nn.forward(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        checkpoints = info['checkpoints_visited']
        
        if info['collision']:
            collisions += 1
        
        if terminated:
            print(f"Episode terminated at step {step + 1}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print(f"Checkpoints visited: {checkpoints}")
    print(f"Collisions: {collisions}")
    
    summary = env.episode_data.get_summary()
    print(f"Episode summary: {summary}")
    
    print("✓ Environment episode test PASSED")


def test_ga_trainer_initialization():
    """Test GA trainer initialization."""
    print("\n" + "="*60)
    print("TEST 4: GA Trainer Initialization")
    print("="*60)
    
    script_dir = os.path.dirname(__file__)
    config = load_config(os.path.join(script_dir, "config", "config.yaml"))
    map_data = MapLoader.load_map(os.path.join(script_dir, "maps", "learning_maze_simple.json"))
    
    # Reduce for testing
    config['GA_PARAMS']['population_size'] = 10
    config['GA_PARAMS']['num_generations'] = 2
    config['GA_PARAMS']['fitness_evaluations'] = 1  # Only 1 eval for speed
    
    print(f"Creating GA trainer...")
    trainer = GATrainer(config, map_data, output_dir="test_results")
    
    print(f"Genome size: {trainer.genome_size}")
    print(f"Population size: {trainer.population_size}")
    print(f"Generations: {trainer.num_generations}")
    
    print("✓ GA Trainer initialization test PASSED")


def test_ga_short_training():
    """Test GA training for a few generations."""
    print("\n" + "="*60)
    print("TEST 5: GA Short Training Run")
    print("="*60)
    
    script_dir = os.path.dirname(__file__)
    config = load_config(os.path.join(script_dir, "config", "config.yaml"))
    map_data = MapLoader.load_map(os.path.join(script_dir, "maps", "learning_maze_simple.json"))
    
    # Reduce for testing
    config['GA_PARAMS']['population_size'] = 8
    config['GA_PARAMS']['num_generations'] = 3
    config['GA_PARAMS']['fitness_evaluations'] = 1
    
    print(f"Training GA for {config['GA_PARAMS']['num_generations']} generations...")
    trainer = GATrainer(config, map_data, output_dir="test_results")
    
    best_genome, best_fitness = trainer.train(verbose=True)
    
    print(f"\nBest genome shape: {best_genome.shape}")
    print(f"Best fitness: {best_fitness:.2f}")
    
    stats = trainer.get_statistics()
    print(f"Total generations trained: {stats['total_generations']}")
    print(f"Final average population fitness: {stats['final_avg_population_fitness']:.2f}")
    
    # Save results
    trainer.save_best_genome("test_best_genome.pkl")
    trainer.save_statistics("test_stats.pkl")
    
    print("✓ GA training test PASSED")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# AGV GA ROBOT - PHASE 2 TEST SUITE")
    print("#"*60)
    
    try:
        test_fitness_function()
        test_neural_network_inference()
        test_environment_episode()
        test_ga_trainer_initialization()
        test_ga_short_training()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
