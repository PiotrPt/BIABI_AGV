"""
Simple real-time visualization for AGV GA training.

Single window showing overlayed trajectories:
- Leader (prev gen best) - green
- Top 3 from current gen - blue, cyan, magenta
"""

import os
import sys
import numpy as np
import pygame
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agv_ga_robot.maps.map_loader import MapLoader


class SimpleVisualization:
    """Single-window visualization with overlayed robot trajectories."""
    
    # Colors
    COLOR_BG = (20, 20, 20)
    COLOR_GRID = (50, 50, 50)
    COLOR_OBSTACLE = (100, 100, 100)
    COLOR_LEADER = (0, 255, 0)      # Green - leader
    COLOR_RANK1 = (0, 150, 255)     # Blue - rank 1
    COLOR_RANK2 = (0, 255, 255)     # Cyan - rank 2
    COLOR_RANK3 = (255, 0, 255)     # Magenta - rank 3
    COLOR_CHECKPOINT_UNVISITED = (255, 200, 0)  # Orange
    COLOR_CHECKPOINT_VISITED = (0, 255, 0)      # Green
    COLOR_TEXT = (200, 200, 200)
    COLOR_HEADER = (100, 200, 255)
    
    def __init__(self, 
                 map_data: Dict,
                 config: Dict,
                 window_width: int = 1800,
                 window_height: int = 1100):
        """
        Initialize simple visualization.
        
        Args:
            map_data: Map configuration
            config: Global configuration
            window_width: Window size
            window_height: Window size
        """
        pygame.init()
        
        self.map_data = map_data
        self.config = config
        self.window_width = window_width
        self.window_height = window_height
        
        self.display = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("AGV GA Training - Live Visualization")
        
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 14)
        self.font_medium = pygame.font.Font(None, 18)
        self.font_large = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 11)
        
        # Map dimensions
        self.map_width = map_data['metadata'].get('canvas_width') or map_data['metadata'].get('width', 1200)
        self.map_height = map_data['metadata'].get('canvas_height') or map_data['metadata'].get('height', 800)
        
        # Padding
        self.padding = 40
        self.canvas_width = window_width - 2 * self.padding
        self.canvas_height = window_height - 2 * self.padding - 60  # Leave space for info
        
        # Scaling
        self.scale_x = self.canvas_width / self.map_width
        self.scale_y = self.canvas_height / self.map_height
        
        # State
        self.generation = 0
        self.best_fitness = 0.0
        self.population_avg_fitness = 0.0
        
        # Current trajectories
        self.leader_trajectory = []
        self.rank1_trajectory = []
        self.rank2_trajectory = []
        self.rank3_trajectory = []
        
        # Current stats
        self.leader_stats = {}
        self.top3_stats = [{}, {}, {}]
        
        # Animation state
        self.animation_mode = False
        self.animation_trajectory = []
        self.animation_current_step = 0
        self.animation_robot_angles = []
        self.animation_stats = {}
        
    def map_to_screen(self, map_x: float, map_y: float) -> Tuple[int, int]:
        """Convert map coordinates to screen coordinates."""
        screen_x = int(self.padding + map_x * self.scale_x)
        screen_y = int(self.padding + map_y * self.scale_y)
        return screen_x, screen_y
    
    def draw_map(self) -> None:
        """Draw map elements (obstacles, checkpoints)."""
        # Canvas background
        pygame.draw.rect(self.display, self.COLOR_BG,
                        (self.padding, self.padding,
                         self.canvas_width, self.canvas_height))
        pygame.draw.rect(self.display, self.COLOR_GRID,
                        (self.padding, self.padding,
                         self.canvas_width, self.canvas_height), 2)
        
        # Draw obstacles
        for obstacle in self.map_data['obstacles']:
            points = obstacle['points']
            for i in range(len(points) - 1):
                p1 = self.map_to_screen(points[i][0], points[i][1])
                p2 = self.map_to_screen(points[i+1][0], points[i+1][1])
                pygame.draw.line(self.display, self.COLOR_OBSTACLE, p1, p2, 
                               max(2, int(obstacle.get('width', 5) * self.scale_x)))
        
        # Draw checkpoints
        for cp in self.map_data['checkpoints']:
            pos = self.map_to_screen(cp['x'], cp['y'])
            radius = max(3, int(cp['radius'] * self.scale_x))
            
            # Check if visited (simple heuristic)
            is_visited = False
            color = self.COLOR_CHECKPOINT_VISITED if is_visited else self.COLOR_CHECKPOINT_UNVISITED
            pygame.draw.circle(self.display, color, pos, radius, 2)
    
    def draw_trajectories(self) -> None:
        """Draw all overlayed trajectories with sensors and target arrows."""
        import math
        
        colors_and_trajs = [
            (self.COLOR_LEADER, self.leader_trajectory, self.leader_stats),
            (self.COLOR_RANK1, self.rank1_trajectory, self.top3_stats[0] if len(self.top3_stats) > 0 else {}),
            (self.COLOR_RANK2, self.rank2_trajectory, self.top3_stats[1] if len(self.top3_stats) > 1 else {}),
            (self.COLOR_RANK3, self.rank3_trajectory, self.top3_stats[2] if len(self.top3_stats) > 2 else {}),
        ]
        
        for color, trajectory, stats in colors_and_trajs:
            if len(trajectory) > 1:
                for i in range(len(trajectory) - 1):
                    p1 = self.map_to_screen(trajectory[i][0], trajectory[i][1])
                    p2 = self.map_to_screen(trajectory[i+1][0], trajectory[i+1][1])
                    pygame.draw.line(self.display, color, p1, p2, 2)
        
        # Draw current positions as circles with sensors and target arrows
        positions = [
            (self.COLOR_LEADER, self.leader_trajectory, self.leader_stats),
            (self.COLOR_RANK1, self.rank1_trajectory, self.top3_stats[0] if len(self.top3_stats) > 0 else {}),
            (self.COLOR_RANK2, self.rank2_trajectory, self.top3_stats[1] if len(self.top3_stats) > 1 else {}),
            (self.COLOR_RANK3, self.rank3_trajectory, self.top3_stats[2] if len(self.top3_stats) > 2 else {}),
        ]
        
        for color, trajectory, stats in positions:
            if trajectory:
                pos = self.map_to_screen(trajectory[-1][0], trajectory[-1][1])
                pygame.draw.circle(self.display, color, pos, 6)
                
                # Calculate robot angle from last two trajectory points
                # BUT PREFER stats['robot_angle'] as it's more accurate for current position
                robot_angle = 0.0
                
                # First preference: use stored robot_angle from last step
                if stats and 'robot_angle' in stats:
                    robot_angle = float(stats['robot_angle'])
                # Fallback: calculate from trajectory if we have 2+ points
                elif len(trajectory) >= 2:
                    x1, y1 = trajectory[-2][0], trajectory[-2][1]
                    x2, y2 = trajectory[-1][0], trajectory[-1][1]
                    dx = x2 - x1
                    dy = y2 - y1
                    # Only use if movement is significant (not just noise)
                    if abs(dx) > 0.1 or abs(dy) > 0.1:
                        robot_angle = math.atan2(dy, dx)
                
                # Draw sensors if available
                if stats and 'sensor_readings' in stats:
                    self._draw_sensors(pos, trajectory[-1], stats['sensor_readings'], color, robot_angle)
                
                # Draw target checkpoint arrow if available
                if stats and 'target_checkpoint_idx' in stats and stats['target_checkpoint_idx'] is not None:
                    target_idx = stats['target_checkpoint_idx']
                    if target_idx < len(self.map_data.get('checkpoints', [])):
                        target_cp = self.map_data['checkpoints'][target_idx]
                        self._draw_target_arrow(pos, trajectory[-1], (target_cp['x'], target_cp['y']), color)
    
    def draw_info(self) -> None:
        """Draw information panel with reward breakdown."""
        info_y = self.padding + self.canvas_height + 10
        
        # Header
        header = f"Gen {self.generation} | Best: {self.best_fitness:.0f} | Avg: {self.population_avg_fitness:.0f}"
        header_surf = self.font_large.render(header, True, self.COLOR_HEADER)
        self.display.blit(header_surf, (self.padding, info_y))
        
        # Legend with trajectories
        legend_items = [
            (self.COLOR_LEADER, "Leader"),
            (self.COLOR_RANK1, "Rank 1"),
            (self.COLOR_RANK2, "Rank 2"),
            (self.COLOR_RANK3, "Rank 3"),
        ]
        
        legend_y = info_y + 25
        for i, (color, label) in enumerate(legend_items):
            x = self.padding + i * 200
            pygame.draw.line(self.display, color, (x, legend_y), (x + 20, legend_y), 3)
            label_surf = self.font_small.render(label, True, self.COLOR_TEXT)
            self.display.blit(label_surf, (x + 25, legend_y - 6))
        
        # Reward breakdown for all robots
        self._draw_all_reward_breakdowns(legend_y + 20)
    
    def update_episode(self,
                      leader_trajectory: List[Tuple[float, float]],
                      leader_stats: Dict,
                      top3_trajectories: List[List[Tuple[float, float]]],
                      top3_stats: List[Dict]) -> None:
        """
        Update visualization with new episode data.
        
        Args:
            leader_trajectory: Robot path for leader
            leader_stats: Episode stats for leader
            top3_trajectories: List of 3 trajectories
            top3_stats: List of 3 stats dicts
        """
        self.leader_trajectory = leader_trajectory
        self.leader_stats = leader_stats
        
        self.rank1_trajectory = top3_trajectories[0] if len(top3_trajectories) > 0 else []
        self.rank2_trajectory = top3_trajectories[1] if len(top3_trajectories) > 1 else []
        self.rank3_trajectory = top3_trajectories[2] if len(top3_trajectories) > 2 else []
        
        self.top3_stats = top3_stats
        
        # Render frame
        self.display.fill(self.COLOR_BG)
        self.draw_map()
        self.draw_trajectories()
        self.draw_info()
        
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
    
    def update_generation(self, generation: int, best_fitness: float, avg_fitness: float) -> None:
        """Update generation info."""
        self.generation = generation
        self.best_fitness = best_fitness
        self.population_avg_fitness = avg_fitness
    
    def _draw_sensors(self, screen_pos: Tuple[int, int], robot_map_pos: Tuple[float, float], 
                      sensor_readings: list, color, robot_angle: float = 0.0) -> None:
        """Draw sensor rays from robot position."""
        import math
        
        # Ensure sensor_readings is valid list with 3 elements
        if not sensor_readings:
            sensor_readings = [1.0, 1.0, 1.0]  # Default: no obstacles
        else:
            # Convert to list and ensure length 3
            sensor_readings = list(sensor_readings)
            while len(sensor_readings) < 3:
                sensor_readings.append(1.0)
            sensor_readings = sensor_readings[:3]  # Truncate if > 3
        
        # Sensor angles (from config, typically -60, 0, 60 degrees)
        sensor_angles = [-60, 0, 60]  # degrees
        max_sensor_distance = 200  # from config
        
        rx, ry = robot_map_pos
        
        for i, (angle_offset, reading) in enumerate(zip(sensor_angles, sensor_readings)):
            try:
                # Convert reading to float - may be numpy type or None
                reading_val = float(reading) if reading is not None else 1.0
                
                # Clamp to [0, 1] range
                # 0 = obstacle very close, 1 = no obstacle/far
                reading_val = max(0.0, min(1.0, reading_val))
                
                # Distance scales with reading
                # 0 → 1px (obstacle very close)
                # 1 → 200px (no obstacle detected)
                distance = max(1.0, reading_val * max_sensor_distance)
                
                # Calculate sensor ray direction in world coordinates
                # robot_angle is in radians
                sensor_angle_rad = robot_angle + math.radians(angle_offset)
                
                # Calculate end position in map coordinates
                end_x = rx + distance * math.cos(sensor_angle_rad)
                end_y = ry + distance * math.sin(sensor_angle_rad)
                
                end_screen = self.map_to_screen(end_x, end_y)
                
                # Draw sensor ray - RED color
                sensor_color = (255, 0, 0)  # Red
                pygame.draw.line(self.display, sensor_color, screen_pos, end_screen, 1)
                pygame.draw.circle(self.display, sensor_color, end_screen, 2)
            except Exception as e:
                pass  # Silently skip on error
    
    def _draw_target_arrow(self, screen_pos: Tuple[int, int], robot_map_pos: Tuple[float, float],
                          target_map_pos: Tuple[float, float], color) -> None:
        """Draw arrow from robot to target checkpoint."""
        import math
        
        target_screen = self.map_to_screen(target_map_pos[0], target_map_pos[1])
        
        # Draw line from robot to target
        arrow_color = tuple(int(c * 0.7) for c in color)
        pygame.draw.line(self.display, arrow_color, screen_pos, target_screen, 2)
        
        # Calculate direction for arrowhead
        dx = target_screen[0] - screen_pos[0]
        dy = target_screen[1] - screen_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            # Arrowhead size
            arrow_size = 8
            angle = math.atan2(dy, dx)
            
            # Arrowhead points
            p1 = (target_screen[0] - arrow_size * math.cos(angle - math.pi/6),
                  target_screen[1] - arrow_size * math.sin(angle - math.pi/6))
            p2 = (target_screen[0] - arrow_size * math.cos(angle + math.pi/6),
                  target_screen[1] - arrow_size * math.sin(angle + math.pi/6))
            
            # Draw arrowhead
            pygame.draw.polygon(self.display, arrow_color, [target_screen, p1, p2])
    
    def _draw_all_reward_breakdowns(self, start_y: int) -> None:
        """Draw reward breakdown for all 4 robots in columns."""
        all_stats = [
            ("Leader", self.leader_stats),
            ("Rank 1", self.top3_stats[0] if len(self.top3_stats) > 0 else {}),
            ("Rank 2", self.top3_stats[1] if len(self.top3_stats) > 1 else {}),
            ("Rank 3", self.top3_stats[2] if len(self.top3_stats) > 2 else {}),
        ]
        
        x_positions = [
            self.padding,
            self.padding + 400,
            self.padding + 800,
            self.padding + 1200,
        ]
        
        font_tiny = pygame.font.Font(None, 11)  # Smaller font for breakdown
        
        for col, (label, stats) in enumerate(all_stats):
            if col >= len(x_positions):
                break
            
            x = x_positions[col]
            y = start_y
            max_y = self.window_height - 15  # Leave margin at bottom
            
            # Column header
            header_surf = self.font_small.render(label, True, self.COLOR_HEADER)
            self.display.blit(header_surf, (x, y))
            y += 14
            
            if y > max_y:
                continue
            
            # Total reward
            total_reward = stats.get('total_reward', 0)
            total_text = f"Total: {total_reward:.0f}"
            total_surf = font_tiny.render(total_text, True, (0, 255, 0))
            self.display.blit(total_surf, (x, y))
            y += 11
            
            if y > max_y:
                continue
            
            # Reward breakdown
            if stats and 'reward_breakdown' in stats:
                breakdown = stats['reward_breakdown']
                
                rewards_to_show = [
                    ('CP', breakdown.get('checkpoint_reached', 0)),
                    ('Dist', breakdown.get('distance_improvement', 0)),
                    ('Fwd', breakdown.get('forward_bonus', 0)),
                    ('Col', breakdown.get('collision_penalty', 0)),
                    ('IdleR', breakdown.get('idle_rotation_penalty', 0)),
                    ('Sharp', breakdown.get('sharp_rotation_penalty', 0)),
                    ('Stuck', breakdown.get('stuck_penalty', 0)),
                ]
                
                for text_label, value in rewards_to_show:
                    if value != 0 and y < max_y:
                        color = (100, 255, 100) if value >= 0 else (255, 100, 100)
                        text = f"{text_label}:{value:+.0f}"
                        text_surf = font_tiny.render(text, True, color)
                        self.display.blit(text_surf, (x, y))
                        y += 10
    
    def _draw_reward_breakdown(self, start_y: int) -> None:
        """Draw reward breakdown from leader stats."""
        if not self.leader_stats or 'reward_breakdown' not in self.leader_stats:
            return
        
        breakdown = self.leader_stats['reward_breakdown']
        
        # Format reward breakdown for display
        rewards_to_show = [
            ('Checkpoint', breakdown.get('checkpoint_reached', 0)),
            ('Distance', breakdown.get('distance_improvement', 0)),
            ('Forward', breakdown.get('forward_bonus', 0)),
            ('Collision', breakdown.get('collision_penalty', 0)),
            ('Idle Rot', breakdown.get('idle_rotation_penalty', 0)),
            ('Sharp Rot', breakdown.get('sharp_rotation_penalty', 0)),
            ('Stuck', breakdown.get('stuck_penalty', 0)),
        ]
        
        x = self.padding
        y = start_y
        line_height = 18
        
        label_surf = self.font_medium.render("Reward Breakdown:", True, self.COLOR_HEADER)
        self.display.blit(label_surf, (x, y))
        y += line_height
        
        for label, value in rewards_to_show:
            if value != 0:
                color = self.COLOR_TEXT if value >= 0 else (255, 100, 100)  # Red for penalties
                text = f"  {label}: {value:+.0f}"
                text_surf = self.font_small.render(text, True, color)
                self.display.blit(text_surf, (x, y))
                y += line_height
                if y > self.window_height - 20:  # Stop if we run out of space
                    break
    
    def draw_animated_trajectory(self, 
                                 trajectory: List[Tuple[float, float]],
                                 robot_angles: List[float],
                                 stats: Dict,
                                 playback_speed: float = 1.0) -> bool:
        """
        Draw trajectory animation step-by-step (realtime playback).
        
        Args:
            trajectory: Full trajectory as list of (x, y) tuples
            robot_angles: Robot angle at each step in radians
            stats: Episode statistics
            playback_speed: Speed multiplier (1.0 = normal, 2.0 = 2x faster)
        
        Returns:
            True to continue animation, False to stop (user closed window)
        """
        import math
        
        # Initialize animation if not already done
        if not self.animation_mode:
            self.animation_mode = True
            self.animation_trajectory = trajectory
            self.animation_robot_angles = robot_angles if robot_angles else [0.0] * len(trajectory)
            self.animation_stats = stats
            self.animation_current_step = 0
        
        # Check events
        if not self.handle_events():
            return False
        
        # Increment step based on playback speed
        self.animation_current_step += playback_speed
        
        # Clamp to trajectory length
        if self.animation_current_step >= len(trajectory):
            self.animation_current_step = len(trajectory) - 1
        
        current_idx = int(self.animation_current_step)
        
        # Draw background
        self.display.fill(self.COLOR_BG)
        self.draw_map()
        
        # Draw animated trajectory (only up to current step)
        if current_idx > 0:
            animated_traj = trajectory[:current_idx+1]
            for i in range(len(animated_traj) - 1):
                p1 = self.map_to_screen(animated_traj[i][0], animated_traj[i][1])
                p2 = self.map_to_screen(animated_traj[i+1][0], animated_traj[i+1][1])
                pygame.draw.line(self.display, self.COLOR_LEADER, p1, p2, 2)
        
        # Draw robot at current position with direction arrow
        if current_idx < len(trajectory):
            robot_pos = trajectory[current_idx]
            robot_angle = self.animation_robot_angles[current_idx] if current_idx < len(self.animation_robot_angles) else 0.0
            
            screen_pos = self.map_to_screen(robot_pos[0], robot_pos[1])
            
            # Draw robot circle
            pygame.draw.circle(self.display, self.COLOR_LEADER, screen_pos, 8)
            
            # Draw direction arrow
            arrow_length = 15
            arrow_end_x = robot_pos[0] + arrow_length * math.cos(robot_angle)
            arrow_end_y = robot_pos[1] + arrow_length * math.sin(robot_angle)
            arrow_end_screen = self.map_to_screen(arrow_end_x, arrow_end_y)
            
            pygame.draw.line(self.display, (100, 255, 100), screen_pos, arrow_end_screen, 3)
        
        # Draw info panel
        info_y = self.padding + self.canvas_height + 10
        
        # Progress
        progress_pct = (current_idx / len(trajectory) * 100) if trajectory else 0
        progress_text = f"Step: {current_idx} / {len(trajectory)} ({progress_pct:.1f}%) | Reward: {stats.get('total_reward', 0):.0f}"
        progress_surf = self.font_large.render(progress_text, True, self.COLOR_HEADER)
        self.display.blit(progress_surf, (self.padding, info_y))
        
        # Statistics
        info_y += 30
        checkpoint_count = stats.get('checkpoints_visited', 0)
        collision_count = stats.get('collision_count', 0)
        stats_text = f"Checkpoints: {checkpoint_count} | Collisions: {collision_count}"
        stats_surf = self.font_small.render(stats_text, True, self.COLOR_TEXT)
        self.display.blit(stats_surf, (self.padding, info_y))
        
        # Help text
        help_text = "Press ESC or Q to stop"
        help_surf = self.font_tiny.render(help_text, True, (150, 150, 150))
        self.display.blit(help_surf, (self.window_width - 250, self.window_height - 25))
        
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
        
        # Return True if animation not finished
        return current_idx < len(trajectory) - 1
    
    def handle_events(self) -> bool:
        """
        Handle pygame events.
        
        Returns:
            True to continue, False to quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
        return True
    
    def close(self) -> None:
        """Clean up pygame."""
        pygame.quit()


# Alias for compatibility
Visualization = SimpleVisualization
