"""Test script for Phase 1 - Robot physics, sensors, and environment."""

import sys
import os

# Add parent directory to path so we can import agv_ga_robot as package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agv_ga_robot.utils import load_config
from agv_ga_robot.maps import MapLoader
from agv_ga_robot.robot import RobotAgent, Sensors, NeuralNetwork
from agv_ga_robot.env import AGVEnvironment
import numpy as np


def test_robot_physics():
    """Test robot physics and movement."""
    print("\n" + "="*60)
    print("TEST 1: Robot Physics & Movement")
    print("="*60)
    
    robot = RobotAgent(100, 100, start_angle=0)
    
    print(f"Initial position: ({robot.x:.2f}, {robot.y:.2f})")
    print(f"Initial angle: {np.degrees(robot.angle_rad):.2f}°")
    
    # Move forward
    robot.set_motors(1.0, 1.0)
    for _ in range(10):
        robot.update(dt=0.016)
    
    print(f"After moving forward: ({robot.x:.2f}, {robot.y:.2f})")
    print(f"Total distance: {robot.total_distance:.2f} pixels")
    
    # Turn left
    robot.set_motors(0.5, 1.0)
    for _ in range(10):
        robot.update(dt=0.016)
    
    print(f"After turning left: ({robot.x:.2f}, {robot.y:.2f})")
    print(f"Angle: {np.degrees(robot.angle_rad):.2f}°")
    
    # Check speed
    robot.set_motors(0.5, -0.5)
    avg_speed = robot.get_average_speed()
    print(f"Current average speed: {avg_speed:.3f}")
    
    # Check stuck
    robot.set_motors(0.02, 0.02)
    is_stuck = robot.is_stuck(threshold=0.05)
    print(f"Is stuck (threshold=0.05): {is_stuck}")
    
    print("✓ Robot physics test PASSED")


def test_sensors():
    """Test sensor raycasting."""
    print("\n" + "="*60)
    print("TEST 2: Sensors & Raycasting")
    print("="*60)
    
    sensors = Sensors(sensor_angles=[-60, 0, 60], max_distance=200)
    
    # Create a simple obstacle
    obstacles = [
        {
            'points': [[500, 100], [500, 200]],
            'width': 15
        }
    ]
    
    # Robot looking at obstacle
    readings = sensors.get_readings(400, 150, robot_angle_rad=0, obstacles=obstacles)
    print(f"Sensor readings (looking at obstacle): {[f'{r:.3f}' for r in readings]}")
    
    # Robot looking away from obstacle
    readings = sensors.get_readings(400, 150, robot_angle_rad=np.pi, obstacles=obstacles)
    print(f"Sensor readings (looking away): {[f'{r:.3f}' for r in readings]}")
    
    # Empty space
    readings = sensors.get_readings(100, 100, robot_angle_rad=0, obstacles=[])
    print(f"Sensor readings (empty space): {[f'{r:.3f}' for r in readings]}")
    
    print("✓ Sensors test PASSED")


def test_neural_network():
    """Test neural network."""
    print("\n" + "="*60)
    print("TEST 3: Neural Network")
    print("="*60)
    
    net = NeuralNetwork([16, 32, 16, 2])
    
    genome_size = net.get_genome_size()
    print(f"Network architecture: 16 -> 32 -> 16 -> 2")
    print(f"Genome size: {genome_size} parameters")
    
    # Get genome
    genome = net.get_genome()
    print(f"Genome shape: {genome.shape}")
    
    # Forward pass
    test_input = np.random.randn(16)
    output = net.forward(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values (should be in [-1, 1]): {output}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Copy test
    net_copy = net.copy()
    print(f"Network copy successful: {np.allclose(net.get_genome(), net_copy.get_genome())}")
    
    print("✓ Neural Network test PASSED")


def test_environment():
    """Test Gymnasium environment."""
    print("\n" + "="*60)
    print("TEST 4: Gymnasium Environment")
    print("="*60)
    
    # Load config and map - use absolute paths relative to this script
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "config", "config.yaml")
    map_path = os.path.join(script_dir, "maps", "learning_maze_simple.json")
    
    config = load_config(config_path)
    map_data = MapLoader.load_map(map_path)
    
    print(f"Map: {map_data['metadata']['name']}")
    print(f"Resolution: {config['SIMULATION']['resolution']}")
    print(f"Episode steps: {config['EPISODE_STEPS']}")
    print(f"Checkpoints: {len(map_data['checkpoints'])}")
    print(f"Obstacles: {len(map_data['obstacles'])}")
    
    # Create environment
    env = AGVEnvironment(config, map_data)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset and run one episode
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Run 50 steps with random actions
    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}: reward={reward:.3f}, checkpoints={info['checkpoints_visited']}")
        
        if terminated:
            print(f"Episode terminated at step {step + 1}")
            break
    
    print(f"Total reward: {total_reward:.3f}")
    print(f"Final observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    summary = env.episode_data.get_summary()
    print(f"Episode summary: {summary}")
    
    print("✓ Environment test PASSED")


def test_collision_detection():
    """Test collision detection."""
    print("\n" + "="*60)
    print("TEST 5: Collision Detection")
    print("="*60)
    
    robot = RobotAgent(100, 100)
    
    # Simple vertical wall
    obstacles = [
        {
            'points': [[150, 50], [150, 150]],
            'width': 20
        }
    ]
    
    # No collision - far from wall
    collision = robot.check_collision_with_obstacles(obstacles)
    print(f"Robot at (100, 100): collision = {collision} (expected: False)")
    
    # Move towards wall
    robot.x = 140
    robot.y = 100
    collision = robot.check_collision_with_obstacles(obstacles)
    print(f"Robot at (140, 100): collision = {collision} (expected: False)")
    
    # Should collide
    robot.x = 155
    robot.y = 100
    collision = robot.check_collision_with_obstacles(obstacles)
    print(f"Robot at (155, 100): collision = {collision} (expected: True)")
    
    print("✓ Collision detection test PASSED")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# AGV GA ROBOT - PHASE 1 TEST SUITE")
    print("#"*60)
    
    try:
        test_robot_physics()
        test_sensors()
        test_neural_network()
        test_collision_detection()
        test_environment()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
