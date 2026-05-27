"""AGV Environment - Gymnasium-compatible environment for robot training."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Tuple, Dict, Any

from ..robot import RobotAgent, Sensors, NeuralNetwork
from ..maps import MapLoader
from ..utils import get_checkpoint_info, EpisodeData


class AGVEnvironment(gym.Env):
    """
    Gymnasium environment for AGV robot training.
    
    State: 16 normalized observations (3 sensors + 3 checkpoints × 4)
    Action: [left_motor, right_motor] from neural network
    Reward: Multi-component (checkpoint, collision, distance, speed, time)
    """
    
    def __init__(self, config: Dict[str, Any], map_data: Dict[str, Any]):
        """
        Initialize environment.
        
        Args:
            config: Configuration dictionary
            map_data: Map data (from MapLoader)
        """
        super().__init__()
        
        self.config = config
        self.map_data = map_data
        
        # Extract robot config
        robot_cfg = config['ROBOT']
        sim_cfg = config['SIMULATION']
        reward_cfg = config['REWARD_WEIGHTS']
        
        self.dt = sim_cfg['dt']
        self.map_width, self.map_height = sim_cfg['resolution']
        self.max_steps = config['EPISODE_STEPS']
        self.checkpoint_radius = config['CHECKPOINT_RADIUS']
        self.checkpoint_mode = config['CHECKPOINT_MODE']
        
        # Get map elements
        self.start_x, self.start_y, self.start_angle = MapLoader.get_robot_start(map_data)
        self.checkpoints = MapLoader.get_checkpoints(map_data)
        self.obstacles = MapLoader.get_obstacles(map_data)
        
        # Initialize robot and sensors
        self.robot = RobotAgent(
            self.start_x, self.start_y, self.start_angle,
            width=robot_cfg['width'],
            height=robot_cfg['height'],
            max_speed=robot_cfg['max_speed']
        )
        
        self.sensors = Sensors(
            sensor_angles=robot_cfg['sensor_angles'],
            max_distance=robot_cfg['sensor_max_distance']
        )
        
        # Reward weights
        self.reward_weights = reward_cfg
        self.stuck_threshold = robot_cfg.get('stuck_threshold', 0.05)
        
        # Episode tracking
        self.episode_data = EpisodeData()
        self.current_step = 0
        self.visited_checkpoints = [False] * len(self.checkpoints)
        self.prev_closest_distance = float('inf')
        self.steps_without_progress = 0  # Track steps with no progress toward goal
        
        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment.
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset robot
        self.robot.reset(self.start_x, self.start_y, self.start_angle)
        self.visited_checkpoints = [False] * len(self.checkpoints)
        self.current_step = 0
        self.steps_without_progress = 0
        self.episode_data = EpisodeData()
        self.prev_closest_distance = self._get_closest_unvisited_checkpoint_distance()
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of the environment.
        
        Args:
            action: [left_motor, right_motor] from neural network
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Set motor speeds
        self.robot.set_motors(action[0], action[1])
        
        # Physics update
        self.robot.update(self.dt)
        
        # Check collision
        collision = self.robot.check_collision_with_obstacles(self.obstacles)
        
        # Check boundary collision (out of map)
        boundary_collision = (self.robot.x < 0 or self.robot.x > self.map_width or
                            self.robot.y < 0 or self.robot.y > self.map_height)
        
        # Check checkpoint visits
        for i, checkpoint in enumerate(self.checkpoints):
            if not self.visited_checkpoints[i]:
                dx = self.robot.x - checkpoint['x']
                dy = self.robot.y - checkpoint['y']
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < self.checkpoint_radius:
                    self.visited_checkpoints[i] = True
        
        # Calculate reward
        reward = self._calculate_reward(collision or boundary_collision)
        
        # Get observation
        obs = self._get_observation()
        
        # Episode termination
        self.current_step += 1
        terminated = collision or boundary_collision or self.current_step >= self.max_steps
        truncated = self.current_step >= self.max_steps
        
        # Info
        info = {
            'collision': collision,
            'boundary_collision': boundary_collision,
            'checkpoints_visited': sum(self.visited_checkpoints),
            'step': self.current_step,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (normalized)."""
        # Get sensor readings
        sensor_readings = self.sensors.get_readings(
            self.robot.x, self.robot.y, self.robot.angle_rad, self.obstacles
        )
        
        # Calculate checkpoint info
        max_distance = math.sqrt(self.map_width**2 + self.map_height**2)
        checkpoint_data = []
        
        # Find next unvisited checkpoint for priority
        next_checkpoint_idx = None
        for i, visited in enumerate(self.visited_checkpoints):
            if not visited:
                next_checkpoint_idx = i
                break
        
        for i, checkpoint in enumerate(self.checkpoints):
            dist, angle, visited = get_checkpoint_info(
                self.robot.x, self.robot.y, self.robot.angle_rad,
                checkpoint['x'], checkpoint['y'],
                self.visited_checkpoints[i],
                max_distance
            )
            checkpoint_data.extend([dist, math.sin(angle), math.cos(angle), float(visited)])
        
        # Add priority signal: 1.0 if this is the next target, 0.0 otherwise
        # (used in "ordered" mode; in "free" mode this can help prioritize unvisited)
        if next_checkpoint_idx is not None:
            priority = 1.0 if self.checkpoint_mode == "ordered" else 0.0
        else:
            priority = 0.0
        
        # Combine into observation (3 + 3*4 + 1 = 16 values)
        obs = np.array(sensor_readings + checkpoint_data + [priority], dtype=np.float32)
        
        # Ensure all values in [-1, 1]
        obs = np.clip(obs, -1.0, 1.0)
        
        return obs
    
    def _calculate_reward(self, collision: bool) -> float:
        """Calculate reward for this step."""
        reward = 0.0
        
        # Collision penalty
        if collision:
            reward += self.reward_weights['collision_penalty']
            self.episode_data.collision_count += 1
        
        # Checkpoint reward - każdy kolejny checkpoint wart coraz więcej!
        # WAŻNE: śledzenie TYLKO ze stanu visited_checkpoints, nie z episode_data!
        checkpoints_now = sum(self.visited_checkpoints)
        checkpoints_before = self.episode_data.checkpoints_visited
        
        if checkpoints_now > checkpoints_before:
            # Nowy checkpoint został osiągnięty!
            # Multyplikator = ilość checkpointów które już osiągnęliśmy (liczący od 1)
            # CP1 (first) = 1x, CP2 (second) = 2x, CP3 (third) = 3x
            checkpoint_multiplier = checkpoints_now  # checkpoints_now jest nową ilością
            checkpoint_reward = self.reward_weights['checkpoint_reached'] * checkpoint_multiplier
            reward += checkpoint_reward
            self.episode_data.checkpoints_visited = checkpoints_now  # Sync z rzeczywistością!
        
        # Distance improvement reward - główna motywacja do ruchu
        closest_dist = self._get_closest_unvisited_checkpoint_distance()
        distance_improvement = max(0, self.prev_closest_distance - closest_dist)
        reward += self.reward_weights['distance_improvement'] * distance_improvement
        self.episode_data.total_distance_improvement += distance_improvement
        self.prev_closest_distance = closest_dist
        
        # Track progress - if no improvement in 50 steps, apply penalty (was 100)
        if distance_improvement > 0.1:  # Made progress
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1
            if self.steps_without_progress > 50:  # No progress for 50+ steps - DRASTYCZNIE!
                reward += self.reward_weights.get('progress_penalty', -25.0)
        
        # Speed bonus (removed/zeroed out - no reward for pointless movement)
        avg_speed = self.robot.get_average_speed()
        reward += self.reward_weights.get('speed_bonus', 0.0) * avg_speed
        self.episode_data.total_speed += avg_speed
        
        # Stuck penalty
        if self.robot.is_stuck(self.stuck_threshold):
            reward += self.reward_weights['stuck_penalty']
            self.episode_data.stuck_count += 1
        
        # Smart motor penalty/bonus logic
        left_motor = self.robot.left_speed
        right_motor = self.robot.right_speed
        motor_diff = abs(left_motor - right_motor)
        
        # Case 1: Robot stoi z całkowicie słabymi motorami = beznadziejny
        if avg_speed < 0.1 and abs(left_motor) < 0.2 and abs(right_motor) < 0.2:
            reward -= 10.0  # Masywna kara za całkowity brak akcji
        
        # Case 2: Robot stoi ale wykonuje ostry zakręt (motor_diff > 0.7) = OK (potrzebny manewr)
        elif avg_speed < 0.1 and motor_diff > 0.7:
            pass  # Dopuszczamy stanie jeśli robi ostry zakręt
        
        # Case 3: Robot stoi i NIE robi ostrych zakrętów = penalty
        elif avg_speed < 0.1 and motor_diff < 0.7:
            reward -= 5.0  # Kara za stanie bez powodu
        
        # Case 4: Robot jedzie z minimalnymi obrotami (motor_diff < 0.15) = BONUS
        elif avg_speed > 0.1 and motor_diff < 0.15:
            reward += self.reward_weights.get('forward_bonus', 4.0)  # Bonus za jazdę prawie prostą
        
        # Case 5: Robot jedzie ale robi zbędne obroty (0.15 < motor_diff < 0.7) = DUŻA PENALTY
        elif avg_speed > 0.1 and 0.15 < motor_diff < 0.7:
            reward += self.reward_weights.get('idle_rotation_penalty', -10.0)  # Drastycznie karamy zbędne obroty
        
        # Case 6: Robot jedzie i robi ostre obroty (motor_diff > 0.7) = mniejsza kara (mogą być konieczne)
        elif avg_speed > 0.1 and motor_diff > 0.7:
            reward += self.reward_weights.get('sharp_rotation_penalty', -3.0)  # Łagodniejsza kara dla ostrych obrotów
        
        # Loitering penalty: kara za stanięcie na odwiedzonym checkpoincie
        if checkpoints_now > 0 and avg_speed < 0.1:
            for i, checkpoint in enumerate(self.checkpoints):
                if self.visited_checkpoints[i]:
                    dx = self.robot.x - checkpoint['x']
                    dy = self.robot.y - checkpoint['y']
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < self.checkpoint_radius * 1.5:
                        reward -= 0.5
                        break
        
        # Record step
        self.episode_data.add_step(
            reward,
            self.robot.x, self.robot.y, self.robot.angle_rad,
            avg_speed,
            self.robot.is_stuck(self.stuck_threshold)
        )
        
        return float(reward)
    
    def _get_closest_unvisited_checkpoint_distance(self) -> float:
        """Get distance to closest unvisited checkpoint."""
        min_dist = float('inf')
        
        for i, checkpoint in enumerate(self.checkpoints):
            if not self.visited_checkpoints[i]:
                dx = self.robot.x - checkpoint['x']
                dy = self.robot.y - checkpoint['y']
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 0.0
    
    def render(self) -> None:
        """Render environment (implemented in Visualization)."""
        pass
    
    def close(self) -> None:
        """Close environment."""
        pass
