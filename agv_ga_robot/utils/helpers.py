"""Helper utilities - config loading, normalization, etc."""

import yaml
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def normalize_observation(sensor_readings: list, checkpoint_data: list, 
                         map_width: float, map_height: float) -> list:
    """
    Normalize observation to [-1, 1] or [0, 1] range.
    
    Args:
        sensor_readings: [front, left, right] sensor readings [0, 1]
        checkpoint_data: List of (distance, angle, visited) for each checkpoint
        map_width: Map width (for distance normalization)
        map_height: Map height
    
    Returns:
        Normalized observation vector [16 values]
    """
    import math
    
    # Max distance = diagonal of map
    max_distance = math.sqrt(map_width**2 + map_height**2)
    
    obs = []
    
    # Sensor readings [0, 1] - already normalized
    obs.extend(sensor_readings)
    
    # Checkpoint data: distance, sin(angle), cos(angle), visited
    for dist, angle, visited in checkpoint_data:
        # Distance normalized [0, 1]
        obs.append(dist / max_distance)
        
        # Angle as sin/cos [-1, 1]
        obs.append(math.sin(angle))
        obs.append(math.cos(angle))
        
        # Visited [0, 1]
        obs.append(float(visited))
    
    # Clip to ensure all values are in valid range
    import numpy as np
    obs = np.clip(obs, -1.0, 1.0)
    
    return obs.tolist()


def get_checkpoint_info(robot_x: float, robot_y: float, robot_angle: float,
                       checkpoint_x: float, checkpoint_y: float,
                       visited: bool, max_distance: float) -> tuple:
    """
    Calculate distance and angle to checkpoint from robot.
    
    Args:
        robot_x, robot_y: Robot position
        robot_angle: Robot orientation (radians)
        checkpoint_x, checkpoint_y: Checkpoint position
        visited: Whether checkpoint already visited
        max_distance: Max distance for normalization
    
    Returns:
        (distance_normalized, angle_relative, visited_bool)
    """
    import math
    
    # If already visited, don't show it to the network
    if visited:
        # Return max distance (1.0) and zero angle so network ignores it
        return (1.0, 0.0, 1.0)
    
    # Vector to checkpoint
    dx = checkpoint_x - robot_x
    dy = checkpoint_y - robot_y
    
    # Distance
    distance = math.sqrt(dx*dx + dy*dy)
    distance_norm = min(1.0, distance / max_distance)
    
    # Angle to checkpoint in world frame
    angle_to_cp = math.atan2(dy, dx)
    
    # Relative angle (in robot frame)
    angle_relative = angle_to_cp - robot_angle
    
    # Normalize angle to [-pi, pi]
    while angle_relative > math.pi:
        angle_relative -= 2 * math.pi
    while angle_relative < -math.pi:
        angle_relative += 2 * math.pi
    
    return (distance_norm, angle_relative, visited)


class EpisodeData:
    """Container for episode statistics."""
    
    def __init__(self):
        self.checkpoints_visited = 0
        self.collision_count = 0
        self.total_distance_improvement = 0.0
        self.total_speed = 0.0
        self.stuck_count = 0
        self.total_reward = 0.0
        self.steps = 0
        self.trajectory = []  # List of (x, y, angle) for visualization
        
        # Breakdown of reward components
        self.reward_breakdown = {
            'checkpoint_reached': 0.0,
            'collision_penalty': 0.0,
            'distance_improvement': 0.0,
            'speed_bonus': 0.0,
            'stuck_penalty': 0.0,
            'forward_bonus': 0.0,
            'idle_rotation_penalty': 0.0,
            'sharp_rotation_penalty': 0.0,
            'progress_penalty': 0.0,
            'loitering_penalty': 0.0,
            'no_action_penalty': 0.0,
        }
        
        # Current robot state for visualization
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_angle = 0.0
        self.sensor_readings = [0.0, 0.0, 0.0]  # 3 distance sensors
        self.target_checkpoint_idx = None  # Index of next target checkpoint
    
    def add_step(self, reward: float, x: float, y: float, angle: float, 
                speed: float, stuck: bool, reward_breakdown: Dict[str, float] = None,
                sensor_readings: list = None, target_checkpoint: int = None):
        """Record a step in the episode."""
        self.total_reward += reward
        self.total_speed += speed
        self.steps += 1
        if stuck:
            self.stuck_count += 1
        self.trajectory.append((x, y, angle))
        
        # Update robot state for visualization (keep last values)
        self.robot_x = x
        self.robot_y = y
        self.robot_angle = angle
        
        # Update sensor readings if provided
        if sensor_readings is not None:
            self.sensor_readings = sensor_readings
        
        # Update target checkpoint
        if target_checkpoint is not None:
            self.target_checkpoint_idx = target_checkpoint
        
        # Accumulate reward breakdown
        if reward_breakdown:
            for key, value in reward_breakdown.items():
                if key in self.reward_breakdown:
                    self.reward_breakdown[key] += value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get episode summary."""
        return {
            'checkpoints_visited': self.checkpoints_visited,
            'collision_count': self.collision_count,
            'total_distance_improvement': self.total_distance_improvement,
            'total_speed': self.total_speed,
            'stuck_count': self.stuck_count,
            'total_reward': self.total_reward,
            'steps': self.steps,
            'avg_speed': self.total_speed / max(1, self.steps),
            'reward_breakdown': self.reward_breakdown,
            'robot_x': self.robot_x,
            'robot_y': self.robot_y,
            'robot_angle': self.robot_angle,
            'sensor_readings': self.sensor_readings,
            'target_checkpoint_idx': self.target_checkpoint_idx,
        }
